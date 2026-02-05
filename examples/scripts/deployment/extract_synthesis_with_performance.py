"""
End-to-end pipeline: PDF/Markdown papers -> Synthesis + Performance extraction.

This script uses the modular SynthesisPerformancePipeline and its components.

Usage:
    # From markdown files (already extracted from PDFs):
    python extract_synthesis_with_performance.py --input-path /path/to/txt_papers --output-path results/

    # From PDF files (extracts text first):
    python extract_synthesis_with_performance.py --input-path /path/to/pdfs --output-path results/ --from-pdf

    # Skip figure extraction (synthesis only):
    python extract_synthesis_with_performance.py --input-path /path/to/txt_papers --output-path results/ --skip-figures

    # Use electrochemistry plot filter instead of default catalysis filter:
    python extract_synthesis_with_performance.py --input-path /path/to/txt_papers --output-path results/ --domain electrochemistry

    # Disable plot filtering (link all plots):
    python extract_synthesis_with_performance.py --input-path /path/to/txt_papers --output-path results/ --no-filter
"""

import argparse
import logging
import os
import warnings
from pathlib import Path

import dspy
from dotenv import load_dotenv

from llm_synthesis.config.plot_filter_config import PlotFilterConfig
from llm_synthesis.data_loader.paper_loader.fs_paper_loader import FSPaperLoader
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
from llm_synthesis.transformers.material_extraction.dspy_extraction import (
    DspyTextExtractor,
    make_dspy_text_extractor_signature,
)
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

# Silence noisy loggers
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
logging.getLogger("pydantic").setLevel(logging.ERROR)
logging.getLogger("LiteLLM").setLevel(logging.ERROR)
logging.getLogger("litellm").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# System prompt for synthesis extraction
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


def get_plot_filter_config(domain: str | None, no_filter: bool) -> PlotFilterConfig:
    """Get plot filter configuration based on domain or no-filter flag.

    Args:
        domain: Domain name ('catalysis', 'electrochemistry', or None for default)
        no_filter: If True, disable all filtering

    Returns:
        PlotFilterConfig instance
    """
    if no_filter:
        return PlotFilterConfig.no_filter()

    if domain == "electrochemistry":
        return PlotFilterConfig.for_electrochemistry()

    # Default: catalysis
    return PlotFilterConfig.for_catalysis()


def create_pipeline(
    gemini_model: str,
    claude_model: str,
    linker_model: str,
    plot_filter_config: PlotFilterConfig,
    skip_figures: bool,
) -> SynthesisPerformancePipeline:
    """Create and configure the synthesis + performance pipeline.

    Args:
        gemini_model: Gemini model for synthesis extraction
        claude_model: Claude model for plot data extraction
        linker_model: Gemini model for series-to-material linking
        plot_filter_config: Configuration for plot filtering
        skip_figures: If True, skip figure/plot extraction

    Returns:
        Configured SynthesisPerformancePipeline
    """
    gemini_key = os.getenv("GEMINI_API_KEY", "")
    if not gemini_key:
        raise ValueError("GEMINI_API_KEY not found in .env")

    # Material extractor
    material_sig = make_dspy_text_extractor_signature(
        instructions=(
            "Extract ALL distinct material compositions that were synthesized and tested in this paper. "
            "IMPORTANT: If the paper studies multiple variants of a material (e.g., different loadings, "
            "dopant concentrations, or preparation conditions), list EACH variant as a separate material. "
            "For example, if a paper studies 1%Ru/CaO, 3%Ru/CaO, and 5%Ru/CaO, list all three — "
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
        gemini_model,
        model_kwargs={"temperature": 0.0, "max_tokens": 8000, "max_retries": 3},
        system_prompt=SYNTHESIS_SYSTEM_PROMPT,
    )
    synthesis_extractor = DspySynthesisExtractor(signature=synthesis_sig, lm=synthesis_lm)

    # Judge
    judge_lm = get_llm_from_name(
        gemini_model,
        model_kwargs={"temperature": 0.1, "max_tokens": 4096},
    )
    judge_sig = make_general_synthesis_judge_signature()
    judge = DspyGeneralSynthesisJudge(signature=judge_sig, lm=judge_lm)

    # Plot extractor and linker (only if not skipping figures)
    plot_extractor = None
    series_linker = None
    linking_judge = None

    if not skip_figures:
        plot_extractor = ClaudeLinePlotDataExtractor(model_name=claude_model)

        linker_lm = dspy.LM(
            f"gemini/{linker_model}",
            temperature=0.0,
            max_tokens=8000,
            api_key=gemini_key,
        )
        series_linker = SeriesMaterialLinker(lm=linker_lm)

        # Linking judge — evaluates linking quality after performance linking
        linking_judge_lm = get_llm_from_name(
            gemini_model,
            model_kwargs={"temperature": 0.1, "max_tokens": 4096},
        )
        linking_judge_sig = make_linking_judge_signature()
        linking_judge = DspyLinkingJudge(
            signature=linking_judge_sig, lm=linking_judge_lm
        )

    return SynthesisPerformancePipeline(
        material_extractor=material_extractor,
        synthesis_extractor=synthesis_extractor,
        judge=judge,
        linking_judge=linking_judge,
        plot_extractor=plot_extractor,
        series_linker=series_linker,
        plot_filter_config=plot_filter_config,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Extract synthesis + performance data from papers."
    )
    parser.add_argument(
        "--input-path",
        type=str,
        required=True,
        help="Directory containing paper text/markdown files (.txt or .md with embedded base64 images)",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="results/synthesis_performance",
        help="Directory to save results",
    )
    parser.add_argument(
        "--from-pdf",
        action="store_true",
        help="Input is PDFs (extract text first with Docling)",
    )
    parser.add_argument(
        "--skip-figures",
        action="store_true",
        help="Skip figure extraction and performance linking",
    )
    parser.add_argument(
        "--domain",
        type=str,
        choices=["catalysis", "electrochemistry"],
        default="catalysis",
        help="Domain for plot filtering (default: catalysis)",
    )
    parser.add_argument(
        "--no-filter",
        action="store_true",
        help="Disable plot filtering (link all plots)",
    )
    parser.add_argument(
        "--claude-model",
        type=str,
        default="claude-sonnet-4-20250514",
        help="Claude model for plot data extraction",
    )
    parser.add_argument(
        "--gemini-model",
        type=str,
        default="gemini-2.0-flash",
        help="Gemini model for synthesis extraction + judge",
    )
    parser.add_argument(
        "--linker-model",
        type=str,
        default="gemini-3-pro-preview",
        help="Gemini model for performance linking (series-to-material matching)",
    )
    args = parser.parse_args()

    # Load environment
    env_path = Path(__file__).resolve().parents[3] / ".env"
    load_dotenv(env_path, override=True)

    # PDF extraction if needed
    input_path = args.input_path
    if args.from_pdf:
        from llm_synthesis.transformers.pdf_extraction import DoclingPDFExtractor

        txt_output = os.path.join(args.output_path, "_extracted_text")
        os.makedirs(txt_output, exist_ok=True)
        logger.info(f"Extracting text from PDFs in {input_path}...")

        pdf_extractor = DoclingPDFExtractor()
        pdf_dir = Path(input_path)
        for pdf_file in sorted(pdf_dir.glob("*.pdf")):
            md_output = os.path.join(txt_output, pdf_file.stem + ".md")
            if os.path.exists(md_output):
                logger.info(f"  Skipping {pdf_file.name} (already extracted)")
                continue
            logger.info(f"  Extracting {pdf_file.name}...")
            with open(pdf_file, "rb") as f:
                markdown_text = pdf_extractor.forward(f.read())
            with open(md_output, "w", errors="replace") as f:
                f.write(markdown_text)
            logger.info(f"    -> {len(markdown_text)} characters")

        input_path = txt_output
        logger.info(f"Text extracted to {txt_output}")

    # Load papers
    loader = FSPaperLoader(data_dir=input_path)
    papers = loader.load()
    logger.info(f"Loaded {len(papers)} papers from {input_path}")

    if not papers:
        logger.warning("No papers found. Check your input path.")
        return

    # Get plot filter configuration
    plot_filter_config = get_plot_filter_config(args.domain, args.no_filter)
    logger.info(f"Plot filter: domain={args.domain}, no_filter={args.no_filter}")

    # Create pipeline
    pipeline = create_pipeline(
        gemini_model=args.gemini_model,
        claude_model=args.claude_model,
        linker_model=args.linker_model,
        plot_filter_config=plot_filter_config,
        skip_figures=args.skip_figures,
    )

    # Process papers
    os.makedirs(args.output_path, exist_ok=True)

    for paper in papers:
        # Skip if already processed
        paper_dir = os.path.join(args.output_path, paper.id)
        if os.path.isdir(paper_dir) and any(
            f.endswith(".json") and f != "performance_mappings.json"
            for f in os.listdir(paper_dir)
        ):
            logger.info(f"Skipping {paper.name} (already processed)")
            continue

        try:
            result = pipeline.process_paper(paper, skip_figures=args.skip_figures)

            if result:
                pipeline.save_results(result, args.output_path)

                # Print summary
                n_perf = len(result.materials_with_performance)
                logger.info(
                    f"  Done: {len(result.materials)} materials, "
                    f"{result.num_plots} plots, "
                    f"{n_perf} materials with performance data"
                )
        except Exception as e:
            logger.error(f"Failed to process {paper.name}: {e}")

    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()
