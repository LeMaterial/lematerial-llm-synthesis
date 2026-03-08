#!/usr/bin/env python3
"""
Performance-only extraction for papers that already have synthesis data.

Processes existing synthesis results and adds performance data by:
1. Loading existing synthesis from result directories
2. Extracting figures from the corresponding PDFs
3. Extracting plot data using Claude VLM
4. Linking performance data to materials
5. Evaluating linking quality
6. Updating result files with performance data

Usage:
    # Add performance to all papers in results_catalysis
    python run_performance_only.py

    # Specify custom results and PDF directories
    python run_performance_only.py --results ../data/results_catalysis \\
        --pdfs ../data/pdf_papers/catalysis_corpus

    # Process only papers missing performance data
    python run_performance_only.py --only-missing
"""
# ==============================================================================
# IMPORTS
# ==============================================================================

import argparse
import json
import logging
import sys
import time
import traceback
import warnings
from pathlib import Path

from dotenv import load_dotenv

from llm_synthesis.config.plot_filter_config import PlotFilterConfig
from llm_synthesis.metrics.judge.linking_judge import (
    DspyLinkingJudge,
    make_linking_judge_signature,
)
from llm_synthesis.transformers.figure_extraction import FigureExtractorMarkdown
from llm_synthesis.transformers.pdf_extraction import MistralPDFExtractor
from llm_synthesis.transformers.performance_linking.\
    series_material_linker import (
    SeriesMaterialLinker,
)
from llm_synthesis.transformers.plot_extraction.claude_extraction.\
    plot_data_extraction import (
    ClaudeLinePlotDataExtractor,
)
from llm_synthesis.utils.dspy_utils import get_llm_from_name
from llm_synthesis.utils.performance_utils import sanitize_filename

# ==============================================================================
# CONFIGURATION
# ==============================================================================

DEFAULT_RESULTS_DIR = "../data/results_catalysis/"
DEFAULT_PDF_DIR = "../data/pdf_papers/catalysis_corpus"

# Models
GEMINI_MODEL = "gemini-3.0-flash"
CLAUDE_MODEL = "claude-sonnet-4-20250514"
LINKER_MODEL = "gemini-3.0-flash"

# ==============================================================================
# SETUP
# ==============================================================================

src_path = Path("../../src").resolve()
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

env_path = Path("../../.env")
load_dotenv(env_path, override=True)

# Silence noisy loggers
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
logging.getLogger("pydantic").setLevel(logging.ERROR)
logging.getLogger("LiteLLM").setLevel(logging.ERROR)
logging.getLogger("litellm").setLevel(logging.ERROR)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("performance_runner")
# ==============================================================================
# SI FILE DETECTION (copied from run_all_papers.py)
# ==============================================================================

SI_PATTERNS = [
    "_SI",
    "-SI",
    "_si",
    "-si",
    "_Supporting",
    "_supporting",
    "_Supplementary",
    "_supplementary",
    "_supp",
    "_Supp",
]


def is_si_file(path: Path) -> bool:
    """Check if a file is a Supplementary Information file."""
    stem = path.stem
    return any(pattern in stem for pattern in SI_PATTERNS)


def find_si_file(main_paper_path: Path) -> Path | None:
    """Find the SI file matching a main paper."""
    parent_dir = main_paper_path.parent
    main_stem = main_paper_path.stem

    for pattern in SI_PATTERNS:
        for ext in [".pdf", ".md", ".txt"]:
            si_path = parent_dir / f"{main_stem}{pattern}{ext}"
            if si_path.exists():
                return si_path
    return None


def load_file_text(path: Path, pdf_extractor: MistralPDFExtractor) -> str:
    """Load text from a PDF, MD, or TXT file."""
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        with open(path, "rb") as f:
            return pdf_extractor.forward(f.read())
    elif suffix in [".md", ".txt"]:
        with open(path, errors="replace") as f:
            return f.read()
    else:
        raise ValueError(f"Unsupported file type: {suffix}")


# ==============================================================================
# INITIALIZE COMPONENTS
# ==============================================================================


def init_components():
    """Initialize components needed for performance extraction."""
    logger.info("Initializing performance extraction components...")

    # PDF extractor
    pdf_extractor = MistralPDFExtractor(structured=False)

    # Figure extractor
    figure_extractor = FigureExtractorMarkdown()

    # Plot data extractor (Claude VLM)
    plot_extractor = ClaudeLinePlotDataExtractor(model_name=CLAUDE_MODEL)

    # Series-material linker
    linker_lm = get_llm_from_name(
        LINKER_MODEL,
        model_kwargs={"temperature": 0.0, "max_tokens": 16000},
    )
    series_linker = SeriesMaterialLinker(lm=linker_lm)

    # Linking judge
    linking_judge_lm = get_llm_from_name(
        GEMINI_MODEL,
        model_kwargs={"temperature": 0.1, "max_tokens": 16000},
    )
    linking_judge_sig = make_linking_judge_signature()
    linking_judge = DspyLinkingJudge(
        signature=linking_judge_sig, lm=linking_judge_lm
    )

    # Plot filter config
    filter_config = PlotFilterConfig.for_catalysis()

    logger.info("Components initialized.")
    return (
        pdf_extractor,
        figure_extractor,
        plot_extractor,
        series_linker,
        linking_judge,
        filter_config,
    )


# ==============================================================================
# LOAD EXISTING SYNTHESIS
# ==============================================================================


def load_existing_synthesis(result_dir: Path) -> list[tuple[str, dict]]:
    """Load existing synthesis data from a result directory.

    Returns:
        List of (material_name, synthesis_dict) tuples
    """
    materials = []

    for json_file in result_dir.glob("*.json"):
        # Skip summary files
        if (
            "linking_summary" in json_file.name
            or "batch_summary" in json_file.name
            or "performance_mappings" in json_file.name
        ):
            continue

        try:
            with open(json_file) as f:
                data = json.load(f)

            if "material" in data and "synthesis" in data:
                materials.append((data["material"], data))
        except Exception as e:
            logger.warning(f"  Failed to load {json_file.name}: {e}")

    return materials


def find_pdf_for_paper(paper_id: str, pdf_dir: Path) -> Path | None:
    """Find the PDF file for a paper ID."""
    # Try exact match first
    for ext in [".pdf", ".md"]:
        pdf_path = pdf_dir / f"{paper_id}{ext}"
        if pdf_path.exists():
            return pdf_path

    # Try partial match (paper_id might be simplified)
    for pdf_file in pdf_dir.glob("*.pdf"):
        if not is_si_file(pdf_file):
            if paper_id in pdf_file.stem or pdf_file.stem in paper_id:
                return pdf_file

    return None


# ==============================================================================
# PROCESS ONE PAPER
# ==============================================================================


def process_paper_performance(
    paper_id: str,
    result_dir: Path,
    pdf_path: Path,
    pdf_extractor,
    figure_extractor,
    plot_extractor,
    series_linker,
    linking_judge,
    filter_config,
) -> dict:
    """Add performance data to an existing paper's synthesis results."""

    paper_start = time.time()

    logger.info(f"{'=' * 70}")
    logger.info(f"PROCESSING PERFORMANCE: {paper_id}")
    logger.info(f"{'=' * 70}")

    # Step 1: Load existing synthesis
    logger.info("Step 1: Loading existing synthesis data...")
    existing_materials = load_existing_synthesis(result_dir)
    material_names = [m[0] for m in existing_materials]
    logger.info(
        f"  Found {len(existing_materials)} materials: {material_names}"
    )

    if not existing_materials:
        raise ValueError("No existing synthesis data found")

    # Step 2: Load paper text (needed for figure extraction context)
    logger.info("Step 2: Loading paper text...")
    paper_text = load_file_text(pdf_path, pdf_extractor)
    logger.info(f"  Main paper: {len(paper_text):,} characters")

    # Load SI if available
    si_text = ""
    si_path = find_si_file(pdf_path)
    if si_path:
        logger.info(f"  Found SI file: {si_path.name}")
        try:
            si_text = load_file_text(si_path, pdf_extractor)
            logger.info(f"  SI text: {len(si_text):,} characters")
        except Exception as e:
            logger.warning(f"  Failed to load SI: {e}")

    full_text = paper_text + "\n\n" + si_text if si_text else paper_text

    # Step 3: Extract figures
    logger.info("Step 3: Extracting figures...")
    figures = figure_extractor.forward(full_text)
    logger.info(f"  Found {len(figures)} figures")

    if not figures:
        logger.warning("  No figures found - skipping performance extraction")
        return {
            "paper_id": paper_id,
            "total_materials": len(existing_materials),
            "total_figures": 0,
            "total_plots_extracted": 0,
            "plots_linked": 0,
            "materials_with_performance": 0,
            "processing_time_seconds": round(time.time() - paper_start, 1),
        }

    # Step 4: Extract plot data
    logger.info(f"Step 4: Extracting data from {len(figures)} figures...")
    plots = []
    plot_figures = []

    for fig in figures:
        try:
            # Create FigureInfoWithPaper for the extractor
            from llm_synthesis.models.figure import FigureInfoWithPaper

            fig_with_paper = FigureInfoWithPaper(
                base64_data=fig.base64_data,
                alt_text=fig.alt_text,
                position=fig.position,
                context_before=fig.context_before,
                context_after=fig.context_after,
                figure_reference=fig.figure_reference,
                figure_class=fig.figure_class,
                quantitative=fig.quantitative,
                paper_text=full_text[:50000],  # Truncate for context
                si_text="",
            )
            extracted = plot_extractor.forward(fig_with_paper)
            if extracted and hasattr(extracted, "series") and extracted.series:
                plots.append(extracted)
                plot_figures.append(fig)
                logger.info(
                    f"    {fig.figure_reference}: "
                    f"{len(extracted.series)} series extracted"
                )
            else:
                logger.info(f"    {fig.figure_reference}: no series data")
        except Exception as e:
            logger.warning(
                f"    {fig.figure_reference}: extraction failed - {e}"
            )

    logger.info(f"  Extracted data from {len(plots)} plots")

    if not plots:
        logger.warning("  No plot data extracted - skipping linking")
        return {
            "paper_id": paper_id,
            "total_materials": len(existing_materials),
            "total_figures": len(figures),
            "total_plots_extracted": 0,
            "plots_linked": 0,
            "materials_with_performance": 0,
            "processing_time_seconds": round(time.time() - paper_start, 1),
        }

    # Step 5: Filter and link plots to materials
    logger.info(
        f"Step 5: Linking {len(plots)} plots to "
        f"{len(material_names)} materials..."
    )

    # Filter plots based on config
    filtered_plots = []
    filtered_figures = []

    for plot, fig in zip(plots, plot_figures):
        x_axis = getattr(plot, "x_axis_label", "") or ""
        y_axis = getattr(plot, "y_axis_label", "") or ""
        x_unit = getattr(plot, "x_axis_unit", "") or ""
        y_unit = getattr(plot, "y_axis_unit", "") or ""

        # Check if x-axis is relevant (temperature, time, etc.)
        x_relevant = filter_config.is_relevant_x_axis(x_axis, x_unit)
        y_relevant = filter_config.is_relevant_y_axis(y_axis, y_unit)

        if not x_relevant:
            logger.info(
                f"  Skipping plot (not_relevant_x): x='{x_axis}' [{x_unit}]"
            )
            continue
        if not y_relevant:
            logger.info(
                f"  Skipping plot (not_relevant_y): y='{y_axis}' [{y_unit}]"
            )
            continue

        # Check if plot has series data
        series = getattr(plot, "series", []) or []
        if not series:
            logger.info("  Skipping plot (no_series)")
            continue

        filtered_plots.append(plot)
        filtered_figures.append(fig)

    logger.info(f"  {len(filtered_plots)} plots after filtering")

    if not filtered_plots:
        logger.warning("  No relevant plots after filtering")
        return {
            "paper_id": paper_id,
            "total_materials": len(existing_materials),
            "total_figures": len(figures),
            "total_plots_extracted": len(plots),
            "plots_linked": 0,
            "materials_with_performance": 0,
            "processing_time_seconds": round(time.time() - paper_start, 1),
        }

    # Link plots to materials
    plot_mappings = []
    all_unmatched = []

    for i, (plot, fig) in enumerate(zip(filtered_plots, filtered_figures)):
        logger.info(
            f"    Linking plot {i} ({fig.figure_reference}, "
            f"{len(plot.series)} series)"
        )
        try:
            mapping = series_linker.forward(
                plot=plot,
                figure=fig,
                material_names=material_names,
                paper_text=full_text[:50000],  # Truncate for context
            )
            if mapping:
                plot_mappings.append(mapping)
                matched = sum(
                    1 for m in mapping.series_mappings if m.material_name
                )
                unmatched = [
                    m.series_name
                    for m in mapping.series_mappings
                    if not m.material_name
                ]
                all_unmatched.extend(unmatched)
                logger.info(f"      Matched: {matched}, Unmatched: {unmatched}")
        except Exception as e:
            logger.warning(f"      Linking failed: {e}")

    logger.info(f"  Linked {len(plot_mappings)} plots")

    # Step 6: Evaluate linking quality
    logger.info("Step 6: Evaluating linking quality...")
    linking_evaluation = None

    if plot_mappings:
        try:
            linking_evaluation = linking_judge.forward(
                material_names=material_names,
                plot_mappings=plot_mappings,
                paper_text=full_text[:30000],
            )
            if linking_evaluation:
                score = getattr(linking_evaluation, "overall_score", None)
                flags = getattr(linking_evaluation, "failure_flags", [])
                logger.info(f"  Linking evaluation score: {score}/5.0")
                if flags:
                    logger.info(f"  Failure flags: {flags}")
        except Exception as e:
            logger.warning(f"  Linking evaluation failed: {e}")

    # Step 7: Aggregate performance data per material
    logger.info("Step 7: Aggregating performance data...")

    from llm_synthesis.models.performance import (
        PerformanceDataPoint,
    )

    material_performance = {name: [] for name in material_names}

    for mapping in plot_mappings:
        for series_mapping in mapping.series_mappings:
            if (
                series_mapping.material_name
                and series_mapping.material_name in material_performance
            ):
                # Create performance data point
                data_point = PerformanceDataPoint(
                    figure_number=mapping.figure_number,
                    series_name=series_mapping.series_name,
                    x_axis_label=mapping.x_axis_label,
                    x_axis_unit=mapping.x_axis_unit,
                    y_axis_label=mapping.y_axis_label,
                    y_axis_unit=mapping.y_axis_unit,
                    data_points=series_mapping.data_points
                    if hasattr(series_mapping, "data_points")
                    else [],
                    confidence=series_mapping.confidence,
                )
                material_performance[series_mapping.material_name].append(
                    data_point
                )

    materials_with_perf = sum(
        1 for perfs in material_performance.values() if perfs
    )
    logger.info(f"  {materials_with_perf} materials have performance data")

    # Step 8: Update result files
    logger.info("Step 8: Updating result files...")

    for material_name, mat_data in existing_materials:
        perf_data = material_performance.get(material_name, [])

        # Update material file
        mat_filename = sanitize_filename(material_name) + ".json"
        mat_path = result_dir / mat_filename

        if perf_data:
            mat_data["performance"] = {
                "material_name": material_name,
                "data_points": [
                    dp.model_dump()
                    if hasattr(dp, "model_dump")
                    else dp.__dict__
                    for dp in perf_data
                ],
            }
        else:
            mat_data["performance"] = None

        # Add linking evaluation
        mat_data["linking_evaluation"] = (
            linking_evaluation.model_dump() if linking_evaluation else None
        )

        with open(mat_path, "w") as f:
            json.dump(mat_data, f, indent=2, default=str)

    # Save plot mappings
    if plot_mappings:
        mappings_path = result_dir / "performance_mappings.json"
        with open(mappings_path, "w") as f:
            json.dump(
                [m.model_dump() for m in plot_mappings],
                f,
                indent=2,
                default=str,
            )

    # Update linking summaries
    processing_time = time.time() - paper_start

    materials_with_perf_list = [
        name for name, perfs in material_performance.items() if perfs
    ]
    materials_without_perf_list = [
        name for name, perfs in material_performance.items() if not perfs
    ]

    base_summary = {
        "paper_id": paper_id,
        "paper_name": paper_id,
        "total_materials": len(existing_materials),
        "materials_with_synthesis": len(existing_materials),
        "materials_with_performance": len(materials_with_perf_list),
        "materials_without_performance": len(materials_without_perf_list),
        "materials_list": material_names,
        "materials_with_performance_list": materials_with_perf_list,
        "materials_without_performance_list": materials_without_perf_list,
        "total_plots_extracted": len(plots),
        "plots_linked": len(plot_mappings),
        "processing_time_seconds": round(processing_time, 1),
    }

    # LLM summary
    llm_summary = {**base_summary}
    llm_summary["linking_evaluation"] = (
        linking_evaluation.model_dump() if linking_evaluation else None
    )

    with open(result_dir / "linking_summary_llm.json", "w") as f:
        json.dump(llm_summary, f, indent=2, default=str)

    # Human summary template
    human_summary = {**base_summary}
    human_summary["linking_evaluation"] = {
        "reasoning": None,
        "scores": {
            k: None
            for k in [
                "material_identity_score",
                "performance_data_correctness_score",
                "completeness_score",
                "format_structure_score",
                "overall_score",
            ]
        },
        "failure_flags": {f"f{i}": None for i in range(1, 10)},
    }

    with open(result_dir / "linking_summary_human.json", "w") as f:
        json.dump(human_summary, f, indent=2, default=str)

    logger.info(f"  Updated {len(existing_materials)} material files")
    logger.info(f"  Completed in {round(processing_time, 1)}s")

    return {
        "paper_id": paper_id,
        "total_materials": len(existing_materials),
        "total_figures": len(figures),
        "total_plots_extracted": len(plots),
        "plots_linked": len(plot_mappings),
        "materials_with_performance": len(materials_with_perf_list),
        "processing_time_seconds": round(processing_time, 1),
    }


# ==============================================================================
# MAIN
# ==============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Add performance data to existing synthesis results.",
    )
    parser.add_argument(
        "--results",
        default=DEFAULT_RESULTS_DIR,
        help=f"Results directory (default: {DEFAULT_RESULTS_DIR})",
    )
    parser.add_argument(
        "--pdfs",
        default=DEFAULT_PDF_DIR,
        help=f"PDF directory (default: {DEFAULT_PDF_DIR})",
    )
    parser.add_argument(
        "--only-missing",
        action="store_true",
        help="Only process papers that have 0 performance data",
    )
    parser.add_argument(
        "--max",
        type=int,
        default=None,
        help="Maximum number of papers to process",
    )
    args = parser.parse_args()

    results_dir = Path(args.results)
    pdf_dir = Path(args.pdfs)

    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        sys.exit(1)

    if not pdf_dir.exists():
        print(f"Error: PDF directory not found: {pdf_dir}")
        sys.exit(1)

    # Find papers to process
    papers_to_process = []

    for paper_dir in sorted(results_dir.iterdir()):
        if not paper_dir.is_dir():
            continue
        if paper_dir.name.startswith("."):
            continue

        # Check if has synthesis
        summary_path = paper_dir / "linking_summary_llm.json"
        if not summary_path.exists():
            continue

        # Check if needs performance
        if args.only_missing:
            with open(summary_path) as f:
                summary = json.load(f)
            if summary.get("materials_with_performance", 0) > 0:
                logger.info(
                    f"Skipping {paper_dir.name} (already has performance data)"
                )
                continue

        # Find PDF
        pdf_path = find_pdf_for_paper(paper_dir.name, pdf_dir)
        if not pdf_path:
            logger.warning(f"Skipping {paper_dir.name} (PDF not found)")
            continue

        papers_to_process.append((paper_dir.name, paper_dir, pdf_path))

    if args.max:
        papers_to_process = papers_to_process[: args.max]

    logger.info(f"Found {len(papers_to_process)} papers to process")
    for paper_id, _, pdf_path in papers_to_process:
        logger.info(f"  - {paper_id} -> {pdf_path.name}")

    if not papers_to_process:
        print("No papers to process.")
        sys.exit(0)

    # Initialize components
    (
        pdf_extractor,
        figure_extractor,
        plot_extractor,
        series_linker,
        linking_judge,
        filter_config,
    ) = init_components()

    # Process each paper
    all_summaries = []
    total_start = time.time()

    for i, (paper_id, result_dir, pdf_path) in enumerate(papers_to_process, 1):
        logger.info(f"\n{'#' * 70}")
        logger.info(f"# PAPER {i}/{len(papers_to_process)}: {paper_id}")
        logger.info(f"{'#' * 70}")

        try:
            summary = process_paper_performance(
                paper_id,
                result_dir,
                pdf_path,
                pdf_extractor,
                figure_extractor,
                plot_extractor,
                series_linker,
                linking_judge,
                filter_config,
            )
            all_summaries.append(summary)
        except Exception as e:
            error_str = str(e).lower()
            if (
                "rate" in error_str
                or "429" in error_str
                or "quota" in error_str
            ):
                logger.error(f"RATE LIMIT - stopping: {e}")
                all_summaries.append(
                    {"paper_id": paper_id, "error": f"RATE_LIMIT: {e}"}
                )
                break
            logger.error(f"FAILED: {e}")
            traceback.print_exc()
            all_summaries.append({"paper_id": paper_id, "error": str(e)})

    # Print summary
    total_time = round(time.time() - total_start, 1)

    print("\n")
    print("=" * 70)
    print("PERFORMANCE EXTRACTION COMPLETE")
    print("=" * 70)
    n_ok = sum(1 for s in all_summaries if "error" not in s)
    print(f"  Papers processed: {n_ok}/{len(papers_to_process)}")
    print(f"  Total time: {total_time}s ({total_time / 60:.1f} min)")
    print()

    for s in all_summaries:
        if "error" in s:
            print(f"  [FAIL] {s['paper_id']}: {s['error']}")
        else:
            print(
                f"  [OK]   {s['paper_id']}: "
                f"{s['total_plots_extracted']} plots, "
                f"{s['materials_with_performance']}/{s['total_materials']} "
                f"with perf, {s['processing_time_seconds']}s"
            )


if __name__ == "__main__":
    main()
