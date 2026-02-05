#!/usr/bin/env python3
"""
Batch runner for Synthesis + Performance Extraction Pipeline.

Processes all 6 PDF papers (01-06) sequentially without manual intervention.
Results are saved to results/<paper_id>/ for each paper.

Usage:
    python run_all_papers.py
"""

import json
import logging
import os
import sys
import time
import traceback
import warnings
from pathlib import Path

# ==============================================================================
# CONFIGURATION
# ==============================================================================

PDF_DIR = "../data/pdf_papers"
OUTPUT_DIR = "results/"

# Papers to process (in order)
PAPERS = [
    "01_Duan_2012_MCM.pdf",
    "02_Itoh_2002_Hydrogen.pdf",
    "03_Usman_2025_Rare.pdf",
    "04_Wu_2020_Ammonia.pdf",
    "05_Sayas_2020_High.pdf",
    "06_Khan_2023_Recent.pdf",
]

# Models
GEMINI_MODEL = "gemini-2.0-flash"
CLAUDE_MODEL = "claude-sonnet-4-20250514"
LINKER_MODEL = "gemini-3-pro-preview"

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
# IMPORTS (done once)
# ==============================================================================

import dspy

from llm_synthesis.config.plot_filter_config import PlotFilterConfig
from llm_synthesis.models.figure import FigureInfoWithPaper
from llm_synthesis.models.paper import Paper, SynthesisEntry
from llm_synthesis.models.performance import PlotMaterialMapping
from llm_synthesis.metrics.judge.general_synthesis_judge import (
    DspyGeneralSynthesisJudge,
    make_general_synthesis_judge_signature,
)
from llm_synthesis.transformers.figure_extraction import FigureExtractorMarkdown
from llm_synthesis.transformers.material_extraction.dspy_extraction import (
    DspyTextExtractor,
    make_dspy_text_extractor_signature,
)
from llm_synthesis.transformers.pdf_extraction import MistralPDFExtractor
from llm_synthesis.transformers.performance_linking.base import LinkingInput
from llm_synthesis.transformers.performance_linking.plot_filter import PlotFilter
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
from llm_synthesis.metrics.judge.linking_judge import (
    DspyLinkingJudge,
    make_linking_judge_signature,
)
from llm_synthesis.utils import clean_text
from llm_synthesis.utils.dspy_utils import get_llm_from_name
from llm_synthesis.utils.figure_utils import clean_text_from_images
from llm_synthesis.utils.performance_utils import (
    aggregate_all_materials_performance,
    sanitize_filename,
)


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
# INITIALIZE SHARED MODELS (loaded once, reused across papers)
# ==============================================================================

def init_models():
    """Initialize all models once to avoid reloading for each paper."""
    logger.info("Initializing models...")

    # PDF extractor
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
        "gemini-2.5-flash-lite",
        model_kwargs={"temperature": 0.0},
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
        model_kwargs={"temperature": 0.0, "max_tokens": 8000, "max_retries": 3},
        system_prompt=SYNTHESIS_SYSTEM_PROMPT,
    )
    synthesis_extractor = DspySynthesisExtractor(signature=synthesis_sig, lm=synthesis_lm)

    # Judge
    judge_lm = get_llm_from_name(
        GEMINI_MODEL,
        model_kwargs={"temperature": 0.1, "max_tokens": 4096},
    )
    judge_sig = make_general_synthesis_judge_signature()
    judge = DspyGeneralSynthesisJudge(signature=judge_sig, lm=judge_lm)

    # Figure extractor (Florence-2 model)
    figure_extractor = FigureExtractorMarkdown(
        segmenter="florence",
        florence_repo_id="amayuelas/plot-visualization-florence-2-lora-32",
    )

    # Plot data extractor (Claude VLM)
    plot_extractor = ClaudeLinePlotDataExtractor(model_name=CLAUDE_MODEL)

    # Plot filter (catalysis)
    filter_config = PlotFilterConfig.for_catalysis()
    plot_filter = PlotFilter(filter_config)

    # Series-material linker
    gemini_key = os.getenv("GEMINI_API_KEY", "")
    linker_lm = dspy.LM(
        f"gemini/{LINKER_MODEL}",
        temperature=0.0,
        max_tokens=8000,
        api_key=gemini_key,
    )
    series_linker = SeriesMaterialLinker(lm=linker_lm)

    # Linking judge
    linking_judge_lm = get_llm_from_name(
        GEMINI_MODEL,
        model_kwargs={"temperature": 0.1, "max_tokens": 4096},
    )
    linking_judge_sig = make_linking_judge_signature()
    linking_judge = DspyLinkingJudge(
        signature=linking_judge_sig, lm=linking_judge_lm
    )

    logger.info("All models initialized.")

    return {
        "pdf_extractor": pdf_extractor,
        "material_extractor": material_extractor,
        "synthesis_extractor": synthesis_extractor,
        "judge": judge,
        "figure_extractor": figure_extractor,
        "plot_extractor": plot_extractor,
        "plot_filter": plot_filter,
        "series_linker": series_linker,
        "linking_judge": linking_judge,
    }


# ==============================================================================
# PROCESS ONE PAPER
# ==============================================================================

def process_paper(pdf_path: str, models: dict) -> dict:
    """Process a single paper through the full pipeline.

    Returns a summary dict with results and timing.
    """
    paper_start = time.time()
    input_path = Path(pdf_path)
    paper_id = input_path.stem

    logger.info(f"{'=' * 70}")
    logger.info(f"PROCESSING: {input_path.name}")
    logger.info(f"{'=' * 70}")

    # ------------------------------------------------------------------
    # Step 0: Load paper text
    # ------------------------------------------------------------------
    logger.info("Step 0: Extracting text from PDF...")
    if input_path.suffix.lower() == ".pdf":
        with open(input_path, "rb") as f:
            paper_text = models["pdf_extractor"].forward(f.read())
    elif input_path.suffix.lower() in [".md", ".txt"]:
        with open(input_path, "r", errors="replace") as f:
            paper_text = f.read()
    else:
        raise ValueError(f"Unsupported file type: {input_path.suffix}")

    paper = Paper(
        name=paper_id,
        id=paper_id,
        publication_text=paper_text,
        si_text="",
    )
    logger.info(f"  Extracted {len(paper_text):,} characters")

    # ------------------------------------------------------------------
    # Step 1: Extract materials
    # ------------------------------------------------------------------
    logger.info("Step 1: Extracting materials...")
    text_for_llm = clean_text(paper.publication_text)
    materials_text = models["material_extractor"].forward(input=text_for_llm)
    materials = [
        m.strip()
        for m in materials_text.replace("\n", ",").split(",")
        if m.strip()
    ]
    logger.info(f"  Found {len(materials)} materials: {materials}")

    # ------------------------------------------------------------------
    # Step 2: Extract synthesis procedures
    # ------------------------------------------------------------------
    logger.info("Step 2: Extracting synthesis procedures...")
    all_syntheses = []
    for i, material in enumerate(materials, 1):
        logger.info(f"  Synthesis {i}/{len(materials)}: {material}")
        try:
            synthesis = models["synthesis_extractor"].forward(
                input=(text_for_llm, material)
            )
            try:
                evaluation = models["judge"].forward(
                    (text_for_llm, json.dumps(synthesis.model_dump()), material)
                )
                score = evaluation.scores.overall_score
                logger.info(f"    Score: {score}/5.0 | Method: {synthesis.synthesis_method}")
            except Exception as e:
                logger.warning(f"    Judge failed: {e}")
                evaluation = None

            all_syntheses.append(SynthesisEntry(
                material=material,
                synthesis=synthesis,
                evaluation=evaluation,
            ))
        except Exception as e:
            logger.error(f"    Extraction failed: {e}")
            all_syntheses.append(SynthesisEntry(
                material=material, synthesis=None, evaluation=None
            ))

    # ------------------------------------------------------------------
    # Step 3: Extract figures
    # ------------------------------------------------------------------
    logger.info("Step 3: Extracting figures...")
    figures = models["figure_extractor"].forward(paper.publication_text)
    logger.info(f"  Found {len(figures)} subfigures")

    # ------------------------------------------------------------------
    # Step 4: Extract plot data
    # ------------------------------------------------------------------
    logger.info("Step 4: Extracting plot data with Claude VLM...")
    plots = []
    plot_figures = []

    for i, fig in enumerate(figures):
        fig_ref = fig.figure_reference or f"Figure {i}"
        logger.info(f"  Processing {i + 1}/{len(figures)}: {fig_ref}")

        fig_with_paper = FigureInfoWithPaper(
            base64_data=fig.base64_data,
            alt_text=fig.alt_text,
            position=fig.position,
            context_before=fig.context_before,
            context_after=fig.context_after,
            figure_reference=fig.figure_reference,
            figure_class=fig.figure_class,
            quantitative=fig.quantitative,
            paper_text=clean_text_from_images(paper.publication_text),
            si_text=paper.si_text,
        )

        try:
            plot_data = models["plot_extractor"].forward(fig_with_paper)
            if plot_data and plot_data.name_to_coordinates:
                plots.append(plot_data)
                plot_figures.append(fig)
                series = list(plot_data.name_to_coordinates.keys())
                logger.info(f"    Extracted {len(series)} series: {series}")
            else:
                logger.info(f"    No extractable data")
        except Exception as e:
            logger.error(f"    Failed: {e}")

    logger.info(f"  Extracted data from {len(plots)}/{len(figures)} figures")

    # ------------------------------------------------------------------
    # Step 5: Filter plots
    # ------------------------------------------------------------------
    logger.info("Step 5: Filtering plots...")
    if plots:
        relevant_plots, skip_counts = models["plot_filter"].filter_plots(
            plots, log_skipped=True
        )
        logger.info(f"  {len(relevant_plots)} relevant plots out of {len(plots)}")
    else:
        relevant_plots = []

    # ------------------------------------------------------------------
    # Step 6: Link series to materials
    # ------------------------------------------------------------------
    logger.info("Step 6: Linking series to materials...")
    plot_mappings = []

    if relevant_plots:
        for idx, plot in relevant_plots:
            fig = plot_figures[idx]
            series_names = list(plot.name_to_coordinates.keys())
            logger.info(f"  Linking plot {idx}: {len(series_names)} series")

            context = f"{fig.context_before} {fig.context_after}"
            plot_meta = {
                "title": plot.title,
                "x_axis_label": plot.x_axis_label,
                "x_axis_unit": plot.x_axis_unit,
                "y_left_axis_label": plot.y_left_axis_label,
                "y_left_axis_unit": plot.y_left_axis_unit,
            }

            linking_input = LinkingInput(
                materials=materials,
                series_names=series_names,
                context=context,
                plot_metadata=plot_meta,
            )
            validated_mappings = models["series_linker"].forward(linking_input)

            matched_series = {m.series_name for m in validated_mappings}
            unmatched = [s for s in series_names if s not in matched_series]

            plot_mappings.append(PlotMaterialMapping(
                plot_index=idx,
                figure_reference=fig.figure_reference,
                mappings=validated_mappings,
                unmatched_series=unmatched,
            ))

            for m in validated_mappings:
                logger.info(f"    '{m.series_name}' -> '{m.material_name}' ({m.confidence})")
            if unmatched:
                logger.warning(f"    Unmatched: {unmatched}")

    # ------------------------------------------------------------------
    # Step 7: Aggregate performance data
    # ------------------------------------------------------------------
    logger.info("Step 7: Aggregating performance data...")
    if plot_mappings and plots:
        performance_data = aggregate_all_materials_performance(
            materials, plot_mappings, plots
        )
    else:
        performance_data = {}

    # ------------------------------------------------------------------
    # Step 7.5: Evaluate linking quality
    # ------------------------------------------------------------------
    linking_evaluation = None
    if plot_mappings and performance_data:
        logger.info("Step 7.5: Evaluating linking quality...")
        try:
            synthesis_json = json.dumps(
                [
                    {
                        "material": e.material,
                        "synthesis": e.synthesis.model_dump() if e.synthesis else None,
                    }
                    for e in all_syntheses
                ],
                indent=2,
            )
            plot_data_json = json.dumps(
                [p.model_dump() for p in plots], indent=2
            )
            linking_output_json = json.dumps(
                {
                    "mappings": [m.model_dump() for m in plot_mappings],
                    "performance_per_material": {
                        k: v.model_dump() for k, v in performance_data.items()
                    },
                },
                indent=2,
            )
            linking_evaluation = models["linking_judge"].forward(
                (clean_text(paper.publication_text), synthesis_json,
                 plot_data_json, linking_output_json)
            )
            logger.info(
                f"  Linking quality: {linking_evaluation.scores.overall_score}/5.0"
            )
        except Exception as e:
            logger.warning(f"  Linking judge failed: {e}")

    # ------------------------------------------------------------------
    # Step 8: Build final results
    # ------------------------------------------------------------------
    final_results = []
    for entry in all_syntheses:
        result = {
            "material": entry.material,
            "synthesis": entry.synthesis.model_dump() if entry.synthesis else None,
            "evaluation": entry.evaluation.model_dump() if entry.evaluation else None,
            "performance": (
                performance_data[entry.material].model_dump()
                if entry.material in performance_data
                else None
            ),
        }
        final_results.append(result)

    materials_with_perf = [m for m in materials if m in performance_data]
    materials_without_perf = [m for m in materials if m not in performance_data]

    # ------------------------------------------------------------------
    # Step 9: Save results
    # ------------------------------------------------------------------
    logger.info("Step 9: Saving results...")
    paper_dir = os.path.join(OUTPUT_DIR, paper.id)
    os.makedirs(paper_dir, exist_ok=True)

    # Save individual material files (without linking_evaluation)
    for result in final_results:
        mat_name = sanitize_filename(result["material"])
        mat_path = os.path.join(paper_dir, f"{mat_name}.json")
        with open(mat_path, "w") as f:
            json.dump(result, f, indent=2, default=str)

    if plot_mappings:
        mappings_path = os.path.join(paper_dir, "performance_mappings.json")
        with open(mappings_path, "w") as f:
            json.dump([m.model_dump() for m in plot_mappings], f, indent=2)

    # Base summary content
    base_summary = {
        "paper_id": paper.id,
        "paper_name": paper.name,
        "total_materials": len(materials),
        "materials_with_synthesis": sum(1 for r in final_results if r["synthesis"]),
        "materials_with_performance": len(materials_with_perf),
        "materials_without_performance": len(materials_without_perf),
        "materials_list": materials,
        "materials_with_performance_list": materials_with_perf,
        "materials_without_performance_list": materials_without_perf,
        "total_figures": len(figures),
        "total_plots_extracted": len(plots),
        "plots_linked": len(plot_mappings),
        "processing_time_seconds": round(time.time() - paper_start, 1),
    }

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

    logger.info(f"  Saved to {paper_dir}/")
    elapsed = round(time.time() - paper_start, 1)
    logger.info(f"  Completed in {elapsed}s")

    return base_summary


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    total_start = time.time()

    # Check which papers exist
    pdf_dir = Path(PDF_DIR)
    available_papers = []
    for paper_name in PAPERS:
        pdf_path = pdf_dir / paper_name
        # Also check for .md version
        md_path = pdf_dir / paper_name.replace(".pdf", ".md")
        if pdf_path.exists():
            available_papers.append(str(pdf_path))
        elif md_path.exists():
            available_papers.append(str(md_path))
            logger.info(f"Using .md file for {paper_name}")
        else:
            logger.warning(f"Paper not found: {paper_name} (skipping)")

    logger.info(f"Found {len(available_papers)}/{len(PAPERS)} papers to process")

    # Initialize all models once
    models = init_models()

    # Process each paper
    all_summaries = []
    for i, pdf_path in enumerate(available_papers, 1):
        logger.info(f"\n{'#' * 70}")
        logger.info(f"# PAPER {i}/{len(available_papers)}: {Path(pdf_path).name}")
        logger.info(f"{'#' * 70}")

        try:
            summary = process_paper(pdf_path, models)
            all_summaries.append(summary)
        except Exception as e:
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

    batch_path = os.path.join(OUTPUT_DIR, "batch_summary.json")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
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

    print(f"\n  Results: {OUTPUT_DIR}")
    print(f"  Batch summary: {batch_path}")


if __name__ == "__main__":
    main()
