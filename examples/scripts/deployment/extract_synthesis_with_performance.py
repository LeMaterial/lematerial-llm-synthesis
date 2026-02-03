"""
End-to-end pipeline: PDF/Markdown papers -> Synthesis + Performance extraction.

Usage:
    # From markdown files (already extracted from PDFs):
    python extract_synthesis_with_performance.py --input-path /path/to/txt_papers --output-path results/

    # From PDF files (extracts text first):
    python extract_synthesis_with_performance.py --input-path /path/to/pdfs --output-path results/ --from-pdf

    # Skip figure extraction (synthesis only):
    python extract_synthesis_with_performance.py --input-path /path/to/txt_papers --output-path results/ --skip-figures
"""

import argparse
import json
import logging
import os
import warnings
from pathlib import Path

import dspy
from dotenv import load_dotenv

from llm_synthesis.data_loader.paper_loader.fs_paper_loader import FSPaperLoader
from llm_synthesis.metrics.judge.general_synthesis_judge import (
    DspyGeneralSynthesisJudge,
    make_general_synthesis_judge_signature,
)
from llm_synthesis.models.figure import FigureInfo, FigureInfoWithPaper
from llm_synthesis.models.ontologies.general import GeneralSynthesisOntology
from llm_synthesis.models.paper import (
    PaperWithSynthesisOntologies,
    SynthesisEntry,
)
from llm_synthesis.models.plot import ExtractedLinePlotData
from llm_synthesis.transformers.material_extraction.dspy_extraction import (
    DspyTextExtractor,
    make_dspy_text_extractor_signature,
)
from llm_synthesis.transformers.plot_extraction.claude_extraction.plot_data_extraction import (
    ClaudeLinePlotDataExtractor,
)
from llm_synthesis.transformers.synthesis_extraction.dspy_synthesis_extraction import (
    DspySynthesisExtractor,
    make_dspy_synthesis_extractor_signature,
)
from llm_synthesis.utils import clean_text
from llm_synthesis.utils.figure_utils import clean_text_from_images

# Silence noisy loggers
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
logging.getLogger("pydantic").setLevel(logging.ERROR)
logging.getLogger("LiteLLM").setLevel(logging.ERROR)
logging.getLogger("litellm").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# ── Pydantic models for performance linking (same as notebook) ──────────────

from pydantic import BaseModel, Field


class SeriesMapping(BaseModel):
    series_name: str
    material_name: str
    confidence: str = "medium"
    reasoning: str = ""


class PlotMaterialMapping(BaseModel):
    plot_index: int
    figure_reference: str = ""
    mappings: list[SeriesMapping] = Field(default_factory=list)
    unmatched_series: list[str] = Field(default_factory=list)


class MaterialPlotEntry(BaseModel):
    plot_index: int
    figure_reference: str = ""
    series_name: str
    coordinates: list[list[float]] = Field(default_factory=list)
    x_axis_label: str | None = None
    x_axis_unit: str | None = None
    y_axis_label: str | None = None
    y_axis_unit: str | None = None
    plot_title: str | None = None
    confidence: str = "medium"


class MaterialPerformanceData(BaseModel):
    material_name: str
    plot_data: list[MaterialPlotEntry] = Field(default_factory=list)


# ── Performance linking functions (from notebook) ───────────────────────────


def build_matching_prompt(
    materials: list[str],
    series_names: list[str],
    context: str,
    plot_metadata: dict,
) -> str:
    return f"""You are given materials studied in a scientific paper and series/line names
extracted from a plot in that paper. Match each series name to the material
it represents.

Materials: {json.dumps(materials)}
Series names from plot: {json.dumps(series_names)}
Figure context: {context}
Plot title: {plot_metadata.get('title', 'N/A')}
X-axis: {plot_metadata.get('x_axis_label', 'N/A')} ({plot_metadata.get('x_axis_unit', '')})
Y-axis: {plot_metadata.get('y_left_axis_label', 'N/A')} ({plot_metadata.get('y_left_axis_unit', '')})

Return a JSON list of matches. Each match must have:
- "series_name": exactly one of the series names listed above
- "material_name": exactly one of the material names listed above
- "confidence": "high", "medium", or "low"
- "reasoning": brief explanation of why this match was made

Rules:
- Only match when there is a CLEAR connection between the series name and a material name
  (e.g., the names are similar, or context makes the link obvious).
- NEVER guess or assign matches randomly. If you cannot identify a real connection, leave
  the series unmatched. An empty list [] is a valid response.
- If a series is a baseline, reference, substrate, equilibrium line, or pressure condition — do NOT include it.
- series_name and material_name MUST be exactly from the lists above. Do not modify them.

Return ONLY a valid JSON list, no other text."""


def validate_mappings(
    raw_mappings: list[dict],
    valid_series: list[str],
    valid_materials: list[str],
) -> tuple[list[SeriesMapping], list[str]]:
    valid_series_set = set(valid_series)
    valid_materials_set = set(valid_materials)
    validated = []
    matched_series = set()

    for m in raw_mappings:
        sn = m.get("series_name", "")
        mn = m.get("material_name", "")
        if sn not in valid_series_set:
            logger.debug(f"Discarding mapping: series '{sn}' not in plot")
            continue
        if mn not in valid_materials_set:
            logger.debug(f"Discarding mapping: material '{mn}' not in list")
            continue
        validated.append(
            SeriesMapping(
                series_name=sn,
                material_name=mn,
                confidence=m.get("confidence", "medium"),
                reasoning=m.get("reasoning", ""),
            )
        )
        matched_series.add(sn)

    unmatched = [s for s in valid_series if s not in matched_series]
    return validated, unmatched


def match_series_to_materials(
    materials: list[str],
    series_names: list[str],
    context: str,
    plot_metadata: dict,
    lm,
) -> list[dict]:
    prompt = build_matching_prompt(materials, series_names, context, plot_metadata)
    response = lm(prompt)
    response_text = response if isinstance(response, str) else response[0]
    if not response_text:
        logger.warning("LLM returned empty/None response (possibly truncated)")
        return []
    response_text = response_text.strip()
    # Strip markdown code fences
    if response_text.startswith("```"):
        lines = response_text.split("\n")
        response_text = "\n".join(lines[1:-1])
    # Try to extract JSON array from response if there's extra text
    if "[" in response_text:
        start = response_text.index("[")
        end = response_text.rindex("]") + 1
        response_text = response_text[start:end]
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        # Fallback: extract individual JSON objects from malformed array
        import re
        objects = re.findall(r"\{[^{}]+\}", response_text)
        results = []
        for obj_str in objects:
            try:
                results.append(json.loads(obj_str))
            except json.JSONDecodeError:
                continue
        if results:
            logger.debug(f"Recovered {len(results)} mappings from malformed JSON")
            return results
        logger.warning(f"Failed to parse LLM response.\nRaw: {response_text[:300]}")
        return []


def _is_temperature_x_axis(plot: ExtractedLinePlotData) -> bool:
    """Check if a plot has temperature on the x-axis."""
    label = (plot.x_axis_label or "").lower().strip()
    unit = (plot.x_axis_unit or "").lower().strip()
    temp_labels = ["t", "temp", "temperature"]
    # Only match multi-char units to avoid false positives (e.g. 'k' in 'CuKα')
    temp_units = ["°c", "°k", "°f", "ºc", "ºk"]
    return any(t == label for t in temp_labels) or any(u == unit for u in temp_units)


def _is_conversion_y_axis(plot: ExtractedLinePlotData) -> bool:
    """
    Check if a plot has conversion/activity on the y-axis.

    Uses a whitelist approach: only accept plots that clearly indicate
    conversion, yield, selectivity, or similar performance metrics.

    Key insight: Conversion data almost always has '%' as the unit.
    Characterization plots (TCD, XPS, XRD, magnetism, etc.) use
    'a.u.', 'counts', 'emu', 'intensity', etc.
    """
    label = (plot.y_left_axis_label or "").lower().strip()
    unit = (plot.y_left_axis_unit or "").lower().strip()

    # If y-axis is completely empty, we can't verify — reject it
    if not label and not unit:
        return False

    # ══════════════════════════════════════════════════════════════════════
    # Helper: Check if label indicates conversion/performance metric
    # ══════════════════════════════════════════════════════════════════════
    conversion_keywords = [
        "conversion",  # NH3 conversion, CO conversion, etc.
        "yield",       # H2 yield, product yield, etc.
        "selectivity", # N2 selectivity, etc.
        "activity",    # Catalytic activity
    ]
    has_conversion_keyword = any(kw in label for kw in conversion_keywords)

    # Check for "X" as conversion symbol (common in catalysis papers)
    # e.g., "X", "X_NH3", "X_CO", "X (%)" but NOT "XPS", "XRD"
    is_x_conversion = label == "x" or label.startswith("x_") or label.startswith("x ")

    # ══════════════════════════════════════════════════════════════════════
    # MAIN CHECK: % unit is a hint, but must be validated by label
    # ══════════════════════════════════════════════════════════════════════
    # Having % as unit suggests it might be conversion, but labels like
    # "H₂ (%)" or "O₂ (%)" are TPR/TPD characterization plots, not conversion.
    # Require that the label also indicates it's a conversion metric.
    if unit == "%" or unit == "percent":
        if has_conversion_keyword or is_x_conversion:
            return True
        # % unit alone without conversion indicator in label — reject
        return False

    # ══════════════════════════════════════════════════════════════════════
    # SECONDARY CHECK: Label-based detection (even without % unit)
    # ══════════════════════════════════════════════════════════════════════
    if has_conversion_keyword or is_x_conversion:
        return True

    # ══════════════════════════════════════════════════════════════════════
    # DEFAULT: Reject unknown plot types
    # ══════════════════════════════════════════════════════════════════════
    # This avoids including characterization plots like TCD, XPS, XRD,
    # magnetism, impedance, etc. which have varied labels/units
    return False


class LinkingStats(BaseModel):
    """Statistics about plot linking for summary output."""
    total_plots_extracted: int = 0
    plots_linked: int = 0
    plots_skipped_not_temperature_x: int = 0
    plots_skipped_not_conversion_y: int = 0
    plots_skipped_no_series: int = 0
    skipped_plots: list[dict] = Field(default_factory=list)
    linked_plots: list[dict] = Field(default_factory=list)
    all_unmatched_series: list[str] = Field(default_factory=list)
    confidence_counts: dict[str, int] = Field(default_factory=lambda: {"high": 0, "medium": 0, "low": 0})


def link_all_plots(
    materials: list[str],
    plots: list[ExtractedLinePlotData],
    figure_infos: list[FigureInfo],
    lm,
    only_temperature_x: bool = True,
) -> tuple[list[PlotMaterialMapping], LinkingStats]:
    all_mappings = []
    stats = LinkingStats(total_plots_extracted=len(plots))

    for idx, (plot, fig) in enumerate(zip(plots, figure_infos)):
        series_names = list(plot.name_to_coordinates.keys())
        if not series_names:
            stats.plots_skipped_no_series += 1
            continue

        # Filter: only link plots with temperature on x-axis and conversion on y-axis
        if only_temperature_x:
            if not _is_temperature_x_axis(plot):
                logger.info(
                    f"  Skipping plot {idx} '{plot.title or 'N/A'}' "
                    f"(x-axis: '{plot.x_axis_label}' [{plot.x_axis_unit}] — not temperature)"
                )
                stats.plots_skipped_not_temperature_x += 1
                stats.skipped_plots.append({
                    "plot_index": idx,
                    "reason": "not_temperature_x",
                    "x_axis": f"{plot.x_axis_label} [{plot.x_axis_unit}]",
                    "y_axis": f"{plot.y_left_axis_label} [{plot.y_left_axis_unit}]",
                })
                continue
            if not _is_conversion_y_axis(plot):
                logger.info(
                    f"  Skipping plot {idx} '{plot.title or 'N/A'}' "
                    f"(y-axis: '{plot.y_left_axis_label}' [{plot.y_left_axis_unit}] — not conversion)"
                )
                stats.plots_skipped_not_conversion_y += 1
                stats.skipped_plots.append({
                    "plot_index": idx,
                    "reason": "not_conversion_y",
                    "x_axis": f"{plot.x_axis_label} [{plot.x_axis_unit}]",
                    "y_axis": f"{plot.y_left_axis_label} [{plot.y_left_axis_unit}]",
                })
                continue

        context = f"{fig.context_before} {fig.context_after}"
        plot_meta = {
            "title": plot.title,
            "x_axis_label": plot.x_axis_label,
            "x_axis_unit": plot.x_axis_unit,
            "y_left_axis_label": plot.y_left_axis_label,
            "y_left_axis_unit": plot.y_left_axis_unit,
        }

        logger.info(
            f"  Linking plot {idx} '{plot.title or 'N/A'}' "
            f"({len(series_names)} series)"
        )

        raw = match_series_to_materials(materials, series_names, context, plot_meta, lm)
        validated, unmatched = validate_mappings(raw, series_names, materials)

        all_mappings.append(
            PlotMaterialMapping(
                plot_index=idx,
                figure_reference=fig.figure_reference,
                mappings=validated,
                unmatched_series=unmatched,
            )
        )

        # Update stats
        stats.plots_linked += 1
        stats.all_unmatched_series.extend(unmatched)
        for m in validated:
            conf = m.confidence.lower() if m.confidence else "medium"
            if conf in stats.confidence_counts:
                stats.confidence_counts[conf] += 1
        stats.linked_plots.append({
            "plot_index": idx,
            "figure_reference": fig.figure_reference,
            "series_count": len(series_names),
            "matched_count": len(validated),
            "unmatched_count": len(unmatched),
            "y_axis": f"{plot.y_left_axis_label} [{plot.y_left_axis_unit}]",
        })

        logger.info(f"    Matched: {len(validated)}, Unmatched: {unmatched}")

    return all_mappings, stats


def aggregate_performance(
    material_name: str,
    mappings: list[PlotMaterialMapping],
    plots: list[ExtractedLinePlotData],
) -> MaterialPerformanceData:
    entries = []
    for mapping in mappings:
        plot = plots[mapping.plot_index]
        for sm in mapping.mappings:
            if sm.material_name == material_name:
                coords = plot.name_to_coordinates.get(sm.series_name, [])
                entries.append(
                    MaterialPlotEntry(
                        plot_index=mapping.plot_index,
                        figure_reference=mapping.figure_reference,
                        series_name=sm.series_name,
                        coordinates=coords,
                        x_axis_label=plot.x_axis_label,
                        x_axis_unit=plot.x_axis_unit,
                        y_axis_label=plot.y_left_axis_label,
                        y_axis_unit=plot.y_left_axis_unit,
                        plot_title=plot.title,
                        confidence=sm.confidence,
                    )
                )
    return MaterialPerformanceData(material_name=material_name, plot_data=entries)


# ── Main pipeline ───────────────────────────────────────────────────────────


def extract_figures_from_markdown(markdown_text: str) -> list[FigureInfo]:
    """Extract and classify figures from markdown with embedded base64 images."""
    try:
        from llm_synthesis.transformers.figure_extraction.regex_figure_extractor import (
            FigureExtractorMarkdown,
        )

        extractor = FigureExtractorMarkdown()
        all_figures = extractor.forward(markdown_text)

        quantitative_classes = [
            "Bar plots", "Contour plot", "Graph plots",
            "Scatter plot", "Surface plot", "Vector plot",
        ]
        quantitative = [
            f for f in all_figures
            if f.quantitative or f.figure_class in quantitative_classes
        ]
        logger.info(
            f"  Figures: {len(all_figures)} total, "
            f"{len(quantitative)} quantitative"
        )
        return quantitative
    except Exception as e:
        logger.warning(f"  Figure extraction failed: {e}")
        return []


def extract_plot_data(
    figures: list[FigureInfo],
    paper_text: str,
    si_text: str,
    claude_model: str,
) -> tuple[list[ExtractedLinePlotData], list[FigureInfo]]:
    """Extract plot data from quantitative figures using Claude VLM."""
    extractor = ClaudeLinePlotDataExtractor(model_name=claude_model)
    plots = []
    plot_figures = []

    for fig in figures:
        fig_with_paper = FigureInfoWithPaper(
            base64_data=fig.base64_data,
            alt_text=fig.alt_text,
            position=fig.position,
            context_before=fig.context_before,
            context_after=fig.context_after,
            figure_reference=fig.figure_reference,
            figure_class=fig.figure_class,
            quantitative=fig.quantitative,
            paper_text=clean_text_from_images(paper_text),
            si_text=si_text,
        )
        try:
            plot_data = extractor.forward(fig_with_paper)
            if plot_data and plot_data.name_to_coordinates:
                plots.append(plot_data)
                plot_figures.append(fig)
                logger.info(
                    f"    {fig.figure_reference}: "
                    f"{len(plot_data.name_to_coordinates)} series extracted"
                )
        except Exception as e:
            logger.warning(f"    {fig.figure_reference}: extraction failed - {e}")

    return plots, plot_figures


def process_paper(
    paper,
    material_extractor,
    synthesis_extractor,
    judge,
    linker_lm,
    claude_model: str,
    skip_figures: bool,
):
    """Process a single paper through the full pipeline."""
    logger.info(f"Processing: {paper.name}")

    # ── Step 1: Material extraction ─────────────────────────────────────
    text_for_llm = clean_text(paper.publication_text)
    materials_text = material_extractor.forward(input=text_for_llm)

    if not materials_text:
        logger.warning(f"  No materials found, skipping")
        return None

    materials = [
        m.strip()
        for m in materials_text.replace("\n", ",").split(",")
        if m.strip()
    ]
    logger.info(f"  Materials: {materials}")

    # ── Step 2: Synthesis extraction + judge ─────────────────────────────
    all_syntheses = []
    for material in materials:
        logger.info(f"  Extracting synthesis for: {material}")
        try:
            synthesis = synthesis_extractor.forward(
                input=(text_for_llm, material)
            )
            try:
                evaluation = judge.forward(
                    (text_for_llm, json.dumps(synthesis.model_dump()), material)
                )
            except Exception as e:
                logger.warning(f"    Judge failed: {e}")
                evaluation = None

            all_syntheses.append(
                SynthesisEntry(
                    material=material,
                    synthesis=synthesis,
                    evaluation=evaluation,
                )
            )
        except Exception as e:
            logger.error(f"    Synthesis extraction failed: {e}")
            all_syntheses.append(
                SynthesisEntry(
                    material=material,
                    synthesis=GeneralSynthesisOntology(
                        target_compound=material,
                        target_compound_type="other",
                        synthesis_method="other",
                        notes=f"Extraction failed: {e}",
                    ),
                    evaluation=None,
                )
            )

    # ── Step 3: Figure + Plot extraction ────────────────────────────────
    performance_data = {}
    plot_mappings = []
    extracted_plots = []

    if not skip_figures:
        logger.info("  Extracting figures...")
        figures = extract_figures_from_markdown(paper.publication_text)

        if figures:
            logger.info("  Extracting plot data (Claude VLM)...")
            plots, plot_figures = extract_plot_data(
                figures, paper.publication_text, paper.si_text, claude_model
            )
            extracted_plots = plots

    linking_stats = None
    if not skip_figures:
        if plots:
            # ── Step 4: Performance linking ─────────────────────────
            logger.info("  Linking series to materials...")
            plot_mappings, linking_stats = link_all_plots(
                materials, plots, plot_figures, linker_lm
            )

            for mat in materials:
                perf = aggregate_performance(mat, plot_mappings, plots)
                if perf.plot_data:
                    performance_data[mat] = perf

    # ── Step 5: Build results ───────────────────────────────────────────
    results = []
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
        results.append(result)

    # Build summary
    materials_with_perf = [m for m in materials if m in performance_data]
    materials_without_perf = [m for m in materials if m not in performance_data]

    return {
        "paper_id": paper.id,
        "paper_name": paper.name,
        "materials": materials,
        "results": results,
        "plot_mappings": [m.model_dump() for m in plot_mappings],
        "num_plots": len(extracted_plots),
        "linking_stats": linking_stats.model_dump() if linking_stats else None,
        "materials_with_performance": materials_with_perf,
        "materials_without_performance": materials_without_perf,
    }


def _sanitize_filename(name: str) -> str:
    """Turn a material name into a safe filename."""
    return name.replace("/", "_").replace(" ", "_").replace("%", "pct")


def save_results(output: dict, output_dir: str):
    """Save results to disk — one JSON per material + summary."""
    paper_dir = os.path.join(output_dir, output["paper_id"])
    os.makedirs(paper_dir, exist_ok=True)

    # One file per material
    for result in output["results"]:
        mat_name = _sanitize_filename(result["material"])
        mat_path = os.path.join(paper_dir, f"{mat_name}.json")
        with open(mat_path, "w") as f:
            json.dump(result, f, indent=2)

    # Plot mappings (always created, even if empty, for consistency)
    mappings_path = os.path.join(paper_dir, "performance_mappings.json")
    with open(mappings_path, "w") as f:
        json.dump(output["plot_mappings"], f, indent=2)

    # ══════════════════════════════════════════════════════════════════════
    # Summary statistics file for easy evaluation
    # ══════════════════════════════════════════════════════════════════════
    summary = {
        "paper_id": output["paper_id"],
        "paper_name": output["paper_name"],
        "total_materials": len(output["materials"]),
        "materials_with_performance": len(output.get("materials_with_performance", [])),
        "materials_without_performance": len(output.get("materials_without_performance", [])),
        "materials_list": output["materials"],
        "materials_with_performance_list": output.get("materials_with_performance", []),
        "materials_without_performance_list": output.get("materials_without_performance", []),
        "total_plots_extracted": output["num_plots"],
    }

    # Add linking stats if available
    if output.get("linking_stats"):
        stats = output["linking_stats"]
        summary["plots_linked"] = stats.get("plots_linked", 0)
        summary["plots_skipped"] = {
            "not_temperature_x": stats.get("plots_skipped_not_temperature_x", 0),
            "not_conversion_y": stats.get("plots_skipped_not_conversion_y", 0),
            "no_series": stats.get("plots_skipped_no_series", 0),
        }
        summary["confidence_breakdown"] = stats.get("confidence_counts", {})
        summary["all_unmatched_series"] = stats.get("all_unmatched_series", [])
        summary["skipped_plots_detail"] = stats.get("skipped_plots", [])
        summary["linked_plots_detail"] = stats.get("linked_plots", [])

    summary_path = os.path.join(paper_dir, "linking_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(
        f"  Saved {len(output['results'])} material files to {paper_dir}/"
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

    gemini_key = os.getenv("GEMINI_API_KEY", "")
    if not gemini_key:
        raise ValueError("GEMINI_API_KEY not found in .env")

    # ── PDF extraction if needed ────────────────────────────────────────
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

    # ── Load papers ─────────────────────────────────────────────────────
    loader = FSPaperLoader(data_dir=input_path)
    papers = loader.load()
    logger.info(f"Loaded {len(papers)} papers from {input_path}")

    if not papers:
        logger.warning("No papers found. Check your input path.")
        return

    # ── Initialize extractors ───────────────────────────────────────────
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
    from llm_synthesis.utils.dspy_utils import get_llm_from_name

    material_lm = get_llm_from_name(
        "gemini-2.5-flash-lite",
        model_kwargs={"temperature": 0.0},
    )
    material_extractor = DspyTextExtractor(signature=material_sig, lm=material_lm)

    # Synthesis extractor
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

    synthesis_sig = make_dspy_synthesis_extractor_signature(
        instructions=(
            "Extract the complete structured synthesis procedure for the specified material. "
            "Include all steps, conditions (temperature, time, atmosphere), equipment, and precursors. "
            "Be thorough and preserve all quantitative details."
        ),
    )
    synthesis_lm = get_llm_from_name(
        args.gemini_model,
        model_kwargs={"temperature": 0.0, "max_tokens": 8000, "max_retries": 3},
        system_prompt=SYNTHESIS_SYSTEM_PROMPT,
    )
    synthesis_extractor = DspySynthesisExtractor(
        signature=synthesis_sig, lm=synthesis_lm
    )

    # Judge
    judge_lm = get_llm_from_name(
        args.gemini_model,
        model_kwargs={"temperature": 0.1, "max_tokens": 4096},
    )
    judge_sig = make_general_synthesis_judge_signature()
    judge = DspyGeneralSynthesisJudge(signature=judge_sig, lm=judge_lm)

    # Linker LM (for matching series to materials)
    linker_lm = dspy.LM(
        f"gemini/{args.linker_model}",
        temperature=0.0,
        max_tokens=8000,
        api_key=gemini_key,
    )

    # ── Process papers ──────────────────────────────────────────────────
    os.makedirs(args.output_path, exist_ok=True)

    for paper in papers:
        # Skip if already processed (check if paper dir has any .json files)
        paper_dir = os.path.join(args.output_path, paper.id)
        if os.path.isdir(paper_dir) and any(
            f.endswith(".json") and f != "performance_mappings.json"
            for f in os.listdir(paper_dir)
        ):
            logger.info(f"Skipping {paper.name} (already processed)")
            continue

        try:
            output = process_paper(
                paper=paper,
                material_extractor=material_extractor,
                synthesis_extractor=synthesis_extractor,
                judge=judge,
                linker_lm=linker_lm,
                claude_model=args.claude_model,
                skip_figures=args.skip_figures,
            )
            if output:
                save_results(output, args.output_path)

                # Print summary
                n_perf = sum(
                    1 for r in output["results"] if r["performance"] is not None
                )
                logger.info(
                    f"  Done: {len(output['materials'])} materials, "
                    f"{output['num_plots']} plots, "
                    f"{n_perf} materials with performance data"
                )
        except Exception as e:
            logger.error(f"Failed to process {paper.name}: {e}")

    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()
