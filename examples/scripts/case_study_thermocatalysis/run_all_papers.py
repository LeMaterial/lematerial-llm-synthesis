#!/usr/bin/env python3
"""
Batch runner for Synthesis + Performance Extraction Pipeline.

Processes all PDF papers in a folder sequentially without manual intervention.
Results are saved to results/<paper_id>/ for each paper.

Uses the SynthesisPerformancePipeline class for all processing logic,
avoiding code duplication with the notebooks.

Usage:
    # Process all PDFs in the default folder (../data/pdf_papers)
    python run_all_papers.py

    # Process all PDFs in a specific folder
    python run_all_papers.py /path/to/pdf/folder

    # Process all PDFs and save to custom output folder
    python run_all_papers.py /path/to/pdf/folder /path/to/output

    # Process only first 5 papers (testing)
    python run_all_papers.py --max 5

    # Skip papers that already have results
    python run_all_papers.py --skip-existing
"""

import argparse
import json
import logging
import os
import sys
import time
import traceback
import warnings
from pathlib import Path

# ==============================================================================
# CONFIGURATION (defaults, can be overridden via CLI)
# ==============================================================================

DEFAULT_PDF_DIR = "../data/pdf_papers"
DEFAULT_OUTPUT_DIR = "../data/results_catalysis/"

# Models
GEMINI_MODEL = "gemini-3.0-flash"
CLAUDE_MODEL = "claude-sonnet-4-20250514"
LINKER_MODEL = "gemini-3.0-flash"

# ==============================================================================
# SETUP
# ==============================================================================

# Add src directory to Python path
src_path = Path("../../src").resolve()
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from dotenv import load_dotenv

env_path = Path("../../.env")
load_dotenv(env_path, override=True)

# Silence noisy loggers
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
logging.getLogger("pydantic").setLevel(logging.ERROR)
logging.getLogger("LiteLLM").setLevel(logging.ERROR)
logging.getLogger("litellm").setLevel(logging.ERROR)

# Setup logging for this script
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("batch_runner")

# ==============================================================================
# IMPORTS
# ==============================================================================

import dspy

from llm_synthesis.config.plot_filter_config import PlotFilterConfig
from llm_synthesis.models.paper import Paper
from llm_synthesis.metrics.judge.general_synthesis_judge import (
    DspyGeneralSynthesisJudge,
    make_general_synthesis_judge_signature,
)
from llm_synthesis.metrics.judge.linking_judge import (
    DspyLinkingJudge,
    make_linking_judge_signature,
)
from llm_synthesis.services.pipelines.synthesis_performance_pipeline import (
    SynthesisPerformancePipeline,
)
from llm_synthesis.transformers.figure_extraction import FigureExtractorMarkdown
from llm_synthesis.transformers.material_extraction.dspy_extraction import (
    DspyTextExtractor,
    make_dspy_text_extractor_signature,
)
from llm_synthesis.transformers.pdf_extraction import MistralPDFExtractor
from llm_synthesis.transformers.performance_linking.series_material_linker import (
    SeriesMaterialLinker,
)
from llm_synthesis.transformers.plot_extraction.claude_extraction.plot_data_extraction import (
    ClaudeLinePlotDataExtractor,
)
from llm_synthesis.transformers.synthesis_extraction.dspy_synthesis_extraction import (
    DspySynthesisExtractor,
    make_dspy_synthesis_extractor_signature,
)
from llm_synthesis.utils.dspy_utils import get_llm_from_name
from llm_synthesis.utils.performance_utils import sanitize_filename


# ==============================================================================
# SYNTHESIS SYSTEM PROMPT
# ==============================================================================

SYNTHESIS_SYSTEM_PROMPT = """You are a helpful assistant that extracts structured synthesis procedures from scientific papers.

IMPORTANT: For the synthesis_method field, you MUST choose from these exact values:
'PVD', 'CVD', 'arc discharge', 'ball milling', 'spray pyrolysis', 'electrospinning',
'sol-gel', 'hydrothermal', 'solvothermal', 'precipitation', 'coprecipitation', 'combustion',
'microwave-assisted', 'sonochemical', 'template-directed', 'solid-state', 'flux growth',
'float zone & Bridgman', 'arc melting & induction melting', 'spark plasma sintering',
'electrochemical deposition', 'chemical bath deposition', 'liquid-phase epitaxy', 'self-assembly',
'atomic layer deposition', 'molecular beam epitaxy', 'pulsed laser deposition', 'ion implantation',
'lithographic patterning', 'wet impregnation', 'incipient wetness impregnation', 'mechanical mixing',
'solution-based', 'mechanochemical', 'other'

For the target_compound_type field, you MUST choose from these exact values:
'metals & alloys', 'ceramics & glasses', 'polymers & soft matter', 'composites',
'semiconductors & electronic', 'nanomaterials', 'two-dimensional materials',
'framework & porous materials', 'biomaterials & biological', 'liquid materials',
'hybrid & organic-inorganic', 'functional materials & catalysts', 'energy & sustainability',
'smart & responsive materials', 'emerging & quantum materials', 'other'

If the exact method is not in the list, use the closest match or 'other'."""


# ==============================================================================
# INITIALIZE PIPELINE
# ==============================================================================

def init_pipeline() -> tuple[MistralPDFExtractor, SynthesisPerformancePipeline]:
    """Initialize the PDF extractor and synthesis-performance pipeline.

    Returns:
        Tuple of (pdf_extractor, pipeline)
    """
    logger.info("Initializing pipeline components...")

    # PDF extractor (separate from pipeline as it handles raw bytes)
    pdf_extractor = MistralPDFExtractor(structured=False)

    # Material extractor
    material_sig = make_dspy_text_extractor_signature(
        instructions=(
            "Extract ALL distinct material compositions that were synthesized and tested in this paper. "
            "IMPORTANT: If the paper studies multiple variants of a material (e.g., different loadings, "
            "dopant concentrations, or preparation conditions), list EACH variant as a separate material. "
            "For example, if a paper studies 1%Ru/CaO, 3%Ru/CaO, and 5%Ru/CaO, list all three - "
            "do NOT merge them into a single 'Ru/CaO'. "
            "Focus on materials that were actually synthesized, not just mentioned or referenced."
        ),
        output_description=(
            "ALL distinct synthesized material compositions as a comma-separated list using chemical formulas. "
            "Include loading percentages and promoters when specified "
            "(e.g., '1%Ru-10%K/CaO, 3%Ru-10%K/CaO, 5%Ru-10%K/CaO, 3%Ru-5%K/CaO'). "
            "Never merge variants into a single generic name."
        ),
    )
    material_lm = get_llm_from_name(
        "gemini-3.0-pro",
        model_kwargs={"temperature": 0.0, "max_tokens": 8000},
    )
    material_extractor = DspyTextExtractor(signature=material_sig, lm=material_lm)

    # Synthesis extractor
    synthesis_sig = make_dspy_synthesis_extractor_signature(
        instructions=(
            "Extract the complete structured synthesis procedure for the specified material. "
            "Include all steps, conditions (temperature, time, atmosphere), equipment, and precursors. "
            "Be thorough and preserve all quantitative details."
        ),
    )
    synthesis_lm = get_llm_from_name(
        GEMINI_MODEL,
        model_kwargs={"temperature": 0.0, "max_tokens": 80000, "max_retries": 3},
        system_prompt=SYNTHESIS_SYSTEM_PROMPT,
    )
    synthesis_extractor = DspySynthesisExtractor(signature=synthesis_sig, lm=synthesis_lm)

    # Synthesis judge
    judge_lm = get_llm_from_name(
        GEMINI_MODEL,
        model_kwargs={"temperature": 0.1, "max_tokens": 20000},
    )
    judge_sig = make_general_synthesis_judge_signature()
    judge = DspyGeneralSynthesisJudge(signature=judge_sig, lm=judge_lm)

    # Plot data extractor (Claude VLM)
    plot_extractor = ClaudeLinePlotDataExtractor(model_name=CLAUDE_MODEL)

    # Series-material linker (increased max_tokens to handle large papers)
    linker_lm = get_llm_from_name(
        LINKER_MODEL,
        model_kwargs={"temperature": 0.0, "max_tokens": 32000},
    )
    series_linker = SeriesMaterialLinker(lm=linker_lm)

    # Linking judge
    linking_judge_lm = get_llm_from_name(
        GEMINI_MODEL,
        model_kwargs={"temperature": 0.1, "max_tokens": 60000},
    )
    linking_judge_sig = make_linking_judge_signature()
    linking_judge = DspyLinkingJudge(
        signature=linking_judge_sig, lm=linking_judge_lm
    )

    # Plot filter config (catalysis)
    filter_config = PlotFilterConfig.for_catalysis()

    # Create the pipeline
    pipeline = SynthesisPerformancePipeline(
        material_extractor=material_extractor,
        synthesis_extractor=synthesis_extractor,
        judge=judge,
        linking_judge=linking_judge,
        plot_extractor=plot_extractor,
        series_linker=series_linker,
        plot_filter_config=filter_config,
    )

    logger.info("Pipeline initialized.")
    return pdf_extractor, pipeline


# ==============================================================================
# CUSTOM SAVE FUNCTION (with human annotation template)
# ==============================================================================

def save_results_with_annotations(result, output_dir: str, processing_time: float) -> None:
    """Save pipeline results with both LLM and human annotation templates.

    Extends the pipeline's save_results with:
    - linking_summary_llm.json: Summary + LLM evaluation
    - linking_summary_human.json: Summary + empty fields for human annotation

    Args:
        result: PipelineResult from the pipeline
        output_dir: Base output directory
        processing_time: Time taken to process this paper (seconds)
    """
    paper_dir = os.path.join(output_dir, result.paper_id)
    os.makedirs(paper_dir, exist_ok=True)

    # Save individual material files (without linking_evaluation)
    for entry in result.results:
        mat_name = sanitize_filename(entry.material)
        mat_path = os.path.join(paper_dir, f"{mat_name}.json")

        # Build dict without linking_evaluation (it goes in summary)
        mat_data = {
            "material": entry.material,
            "synthesis": entry.synthesis.model_dump() if entry.synthesis else None,
            "evaluation": entry.evaluation.model_dump() if entry.evaluation else None,
            "performance": entry.performance.model_dump() if entry.performance else None,
        }
        with open(mat_path, "w") as f:
            json.dump(mat_data, f, indent=2, default=str)

    # Save plot mappings
    if result.plot_mappings:
        mappings_path = os.path.join(paper_dir, "performance_mappings.json")
        with open(mappings_path, "w") as f:
            json.dump([m.model_dump() for m in result.plot_mappings], f, indent=2)

    # Base summary content
    base_summary = {
        "paper_id": result.paper_id,
        "paper_name": result.paper_name,
        "total_materials": len(result.materials),
        "materials_with_synthesis": sum(1 for r in result.results if r.synthesis),
        "materials_with_performance": len(result.materials_with_performance),
        "materials_without_performance": len(result.materials_without_performance),
        "materials_list": result.materials,
        "materials_with_performance_list": result.materials_with_performance,
        "materials_without_performance_list": result.materials_without_performance,
        "total_plots_extracted": result.num_plots,
        "plots_linked": len(result.plot_mappings),
        "processing_time_seconds": round(processing_time, 1),
    }

    # Add linking stats if available
    if result.linking_stats:
        stats = result.linking_stats
        base_summary["plots_skipped"] = {
            "not_relevant_x": stats.plots_skipped_not_relevant_x,
            "not_relevant_y": stats.plots_skipped_not_relevant_y,
            "no_series": stats.plots_skipped_no_series,
        }
        base_summary["confidence_breakdown"] = stats.confidence_counts
        base_summary["all_unmatched_series"] = stats.all_unmatched_series

    # Get linking evaluation from the first result entry (it's the same for all)
    linking_evaluation = None
    if result.results and result.results[0].linking_evaluation:
        linking_evaluation = result.results[0].linking_evaluation

    # linking_summary_llm.json: summary + LLM evaluation
    llm_summary = {**base_summary}
    llm_summary["linking_evaluation"] = (
        linking_evaluation.model_dump() if linking_evaluation else None
    )
    with open(os.path.join(paper_dir, "linking_summary_llm.json"), "w") as f:
        json.dump(llm_summary, f, indent=2, default=str)

    # linking_summary_human.json: summary + empty evaluation for annotation
    human_summary = {**base_summary}
    human_summary["linking_evaluation"] = {
        "reasoning": None,
        "scores": {
            "material_identity_score": None,
            "material_identity_reasoning": None,
            "performance_data_correctness_score": None,
            "performance_data_correctness_reasoning": None,
            "completeness_score": None,
            "completeness_reasoning": None,
            "format_structure_score": None,
            "format_structure_reasoning": None,
            "overall_score": None,
            "overall_reasoning": None,
        },
        "failure_flags": {
            "f1_name_mismatch": None,
            "f2_one_to_many_synthesis": None,
            "f3_many_to_one_figure": None,
            "f4_sample_code_failure": None,
            "f5_precursor_vs_product": None,
            "f6_characterization_confusion": None,
            "f7_dual_axis_error": None,
            "f8_false_negative": None,
            "f9_false_positive": None,
        },
        "confidence_level": None,
        "missing_links": None,
        "spurious_links": None,
        "improvement_suggestions": None,
    }
    with open(os.path.join(paper_dir, "linking_summary_human.json"), "w") as f:
        json.dump(human_summary, f, indent=2, default=str)

    logger.info(f"  Saved {len(result.results)} material files to {paper_dir}/")


# ==============================================================================
# SI FILE DETECTION
# ==============================================================================

# Patterns that indicate a file is supplementary information
SI_PATTERNS = ["_SI", "-SI", "_si", "-si", "_Supporting", "_supporting",
               "_Supplementary", "_supplementary", "_supp", "_Supp"]


def is_si_file(path: Path) -> bool:
    """Check if a file is a Supplementary Information file."""
    stem = path.stem
    return any(pattern in stem for pattern in SI_PATTERNS)


def find_si_file(main_paper_path: Path) -> Path | None:
    """Find the SI file matching a main paper.

    Looks for files like:
    - MainPaper_SI.pdf
    - MainPaper-SI.pdf
    - MainPaper_Supporting.pdf
    - MainPaper_Supplementary.pdf

    Args:
        main_paper_path: Path to the main paper file

    Returns:
        Path to SI file if found, None otherwise
    """
    parent_dir = main_paper_path.parent
    main_stem = main_paper_path.stem

    # Try each SI pattern
    for pattern in SI_PATTERNS:
        # Look for PDF first, then MD
        for ext in [".pdf", ".md", ".txt"]:
            si_path = parent_dir / f"{main_stem}{pattern}{ext}"
            if si_path.exists():
                return si_path

    return None


def load_file_text(path: Path, pdf_extractor: MistralPDFExtractor) -> str:
    """Load text from a PDF, MD, or TXT file.

    Args:
        path: Path to the file
        pdf_extractor: PDF extractor for PDF files

    Returns:
        Extracted text content
    """
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        with open(path, "rb") as f:
            return pdf_extractor.forward(f.read())
    elif suffix in [".md", ".txt"]:
        with open(path, "r", errors="replace") as f:
            return f.read()
    else:
        raise ValueError(f"Unsupported file type: {suffix}")


# ==============================================================================
# PROCESS ONE PAPER
# ==============================================================================

def process_paper(
    pdf_path: str,
    pdf_extractor: MistralPDFExtractor,
    pipeline: SynthesisPerformancePipeline,
    output_dir: str,
) -> dict:
    """Process a single paper through the pipeline.

    Args:
        pdf_path: Path to the PDF or MD file
        pdf_extractor: PDF extractor instance
        pipeline: Initialized SynthesisPerformancePipeline
        output_dir: Output directory for results

    Returns:
        Summary dict with paper statistics
    """
    paper_start = time.time()
    input_path = Path(pdf_path)
    paper_id = input_path.stem

    logger.info(f"{'=' * 70}")
    logger.info(f"PROCESSING: {input_path.name}")
    logger.info(f"{'=' * 70}")

    # Step 0: Load main paper text
    logger.info("Step 0: Extracting text from file...")
    paper_text = load_file_text(input_path, pdf_extractor)
    logger.info(f"  Main paper: {len(paper_text):,} characters")

    # Step 0b: Look for and load SI file
    si_text = ""
    si_path = find_si_file(input_path)
    if si_path:
        logger.info(f"  Found SI file: {si_path.name}")
        try:
            si_text = load_file_text(si_path, pdf_extractor)
            logger.info(f"  SI text: {len(si_text):,} characters")
        except Exception as e:
            logger.warning(f"  Failed to load SI file: {e}")

    paper = Paper(
        name=paper_id,
        id=paper_id,
        publication_text=paper_text,
        si_text=si_text,
    )
    logger.info(f"  Total: {len(paper_text) + len(si_text):,} characters")

    # Run the pipeline
    result = pipeline.process_paper(paper, skip_figures=False)

    if result is None:
        raise ValueError("Pipeline returned no results (no materials found?)")

    processing_time = time.time() - paper_start

    # Save with custom function (adds human annotation template)
    save_results_with_annotations(result, output_dir, processing_time)

    logger.info(f"  Completed in {round(processing_time, 1)}s")

    # Return summary for batch tracking
    return {
        "paper_id": result.paper_id,
        "paper_name": result.paper_name,
        "total_materials": len(result.materials),
        "materials_with_synthesis": sum(1 for r in result.results if r.synthesis),
        "materials_with_performance": len(result.materials_with_performance),
        "materials_without_performance": len(result.materials_without_performance),
        "total_plots_extracted": result.num_plots,
        "plots_linked": len(result.plot_mappings),
        "processing_time_seconds": round(processing_time, 1),
    }


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Batch process PDF papers for synthesis + performance extraction.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_all_papers.py
      Process all PDFs in ../data/pdf_papers

  python run_all_papers.py /path/to/catalysis_papers
      Process all PDFs in the specified folder

  python run_all_papers.py /path/to/papers /path/to/output
      Process PDFs and save results to custom output folder

  python run_all_papers.py --max 5
      Process only the first 5 PDFs (useful for testing)

  python run_all_papers.py --skip-existing
      Skip papers that already have results
        """,
    )
    parser.add_argument(
        "pdf_dir",
        nargs="?",
        default=DEFAULT_PDF_DIR,
        help=f"Directory containing PDF files (default: {DEFAULT_PDF_DIR})",
    )
    parser.add_argument(
        "output_dir",
        nargs="?",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for results (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--max",
        type=int,
        default=None,
        help="Maximum number of papers to process (useful for testing)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip papers that already have results in the output directory",
    )
    args = parser.parse_args()

    pdf_dir = Path(args.pdf_dir)
    output_dir = args.output_dir

    # Validate input directory
    if not pdf_dir.exists():
        print(f"Error: Directory not found: {pdf_dir}")
        sys.exit(1)

    if not pdf_dir.is_dir():
        print(f"Error: Not a directory: {pdf_dir}")
        sys.exit(1)

    total_start = time.time()

    # Find all PDF and MD files in the directory
    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    md_files = sorted(pdf_dir.glob("*.md"))

    # Filter out SI files (they will be loaded automatically with their main paper)
    pdf_files = [p for p in pdf_files if not is_si_file(p)]
    md_files = [p for p in md_files if not is_si_file(p)]

    # Combine: prefer PDF if both exist, otherwise use MD
    pdf_stems = {p.stem for p in pdf_files}
    available_papers = [str(p) for p in pdf_files]
    for md_file in md_files:
        if md_file.stem not in pdf_stems:
            available_papers.append(str(md_file))

    # Sort by filename
    available_papers = sorted(available_papers, key=lambda x: Path(x).name)

    if not available_papers:
        print(f"Error: No PDF or MD files found in {pdf_dir}")
        sys.exit(1)

    # Skip existing if requested
    if args.skip_existing:
        papers_to_process = []
        for paper_path in available_papers:
            paper_id = Path(paper_path).stem
            result_dir = Path(output_dir) / paper_id
            if result_dir.exists() and (result_dir / "linking_summary_llm.json").exists():
                logger.info(f"Skipping {paper_id} (already processed)")
            else:
                papers_to_process.append(paper_path)
        available_papers = papers_to_process

    # Apply max limit if specified
    if args.max and len(available_papers) > args.max:
        logger.info(f"Limiting to first {args.max} papers (--max flag)")
        available_papers = available_papers[: args.max]

    logger.info(f"Found {len(available_papers)} papers to process in {pdf_dir}")
    for p in available_papers:
        logger.info(f"  - {Path(p).name}")

    if not available_papers:
        print("No papers to process.")
        sys.exit(0)

    # Initialize pipeline once
    pdf_extractor, pipeline = init_pipeline()

    # Process each paper
    all_summaries = []
    for i, pdf_path in enumerate(available_papers, 1):
        logger.info(f"\n{'#' * 70}")
        logger.info(f"# PAPER {i}/{len(available_papers)}: {Path(pdf_path).name}")
        logger.info(f"{'#' * 70}")

        try:
            summary = process_paper(pdf_path, pdf_extractor, pipeline, output_dir)
            all_summaries.append(summary)
        except Exception as e:
            error_str = str(e).lower()
            # Stop batch on rate limit errors - don't continue with incomplete data
            if "rate" in error_str and "limit" in error_str or "429" in error_str or "quota" in error_str or "resource_exhausted" in error_str:
                logger.error(f"RATE LIMIT HIT - stopping batch to avoid incomplete data: {e}")
                all_summaries.append({
                    "paper_id": Path(pdf_path).stem,
                    "error": f"RATE_LIMIT: {e}",
                })
                break  # Stop processing more papers
            logger.error(f"FAILED to process {Path(pdf_path).name}: {e}")
            traceback.print_exc()
            all_summaries.append({
                "paper_id": Path(pdf_path).stem,
                "error": str(e),
            })

    # Save overall batch summary
    total_elapsed = round(time.time() - total_start, 1)

    batch_summary = {
        "total_papers": len(available_papers),
        "successful": sum(1 for s in all_summaries if "error" not in s),
        "failed": sum(1 for s in all_summaries if "error" in s),
        "total_time_seconds": total_elapsed,
        "papers": all_summaries,
    }

    batch_path = os.path.join(output_dir, "batch_summary.json")
    os.makedirs(output_dir, exist_ok=True)
    with open(batch_path, "w") as f:
        json.dump(batch_summary, f, indent=2)

    # Print final summary
    print("\n")
    print("=" * 70)
    print("BATCH PROCESSING COMPLETE")
    print("=" * 70)
    print(f"  Papers processed: {batch_summary['successful']}/{batch_summary['total_papers']}")
    print(f"  Failed: {batch_summary['failed']}")
    print(f"  Total time: {total_elapsed}s ({total_elapsed / 60:.1f} min)")
    print()

    for s in all_summaries:
        if "error" in s:
            print(f"  [FAIL] {s['paper_id']}: {s['error']}")
        else:
            print(
                f"  [OK]   {s['paper_id']}: "
                f"{s['total_materials']} materials, "
                f"{s['materials_with_performance']} with perf data, "
                f"{s.get('processing_time_seconds', '?')}s"
            )

    print(f"\n  Results: {output_dir}")
    print(f"  Batch summary: {batch_path}")


if __name__ == "__main__":
    main()
