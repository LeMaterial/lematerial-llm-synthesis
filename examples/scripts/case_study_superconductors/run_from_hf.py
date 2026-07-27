#!/usr/bin/env python3
"""Superconductor case study, reusing text already in the HF dataset.

Skips material + synthesis extraction (both already in
`structured_synthesis` on LeMaterial/LeMat-Synth-Papers,
split=superconductor_keywords_and_LLM) and OCR-derived text metrics.
Only runs the steps that need fresh work per paper:

  1. OCR the PDF -> markdown w/ embedded figure images (Mistral, cached)
  2. Detect + segment figures (Florence-2)
  3. Digitize each plot's series names (Claude), filter to R(T)-relevant
     plots, and link series -> material (DeepSeek, SeriesMaterialLinker)
  4. Read Tc via geometric construction from each linked plot (Qwen via
     LiteLLM), matching to materials via the linker's plot_mappings

materials = ["material_name", ...] read straight from structured_synthesis,
no LLM call.

Usage:
    python run_from_hf.py /path/to/pdf_dir /path/to/output_dir --max 5
"""

import argparse
import json
import logging
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv

_SRC = Path(__file__).resolve().parents[3] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

load_dotenv(Path(__file__).resolve().parents[3] / ".env", override=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DATASET_PATH = (
    "hf://datasets/LeMaterial/LeMat-Synth-Papers/"
    "superconductor_keywords_and_LLM/full-00000-of-00001.parquet"
)

QWEN_MODEL = "qwen3.5-397b-a17b"  # VLM: Tc reading
# Plot digitization: Claude, not Qwen. Tested Qwen via OpenRouter (see
# git history / conversation record) -- it did not produce cleaner series
# names than Claude for doping series (still bare "x=0.05"-style labels,
# sometimes with a "Series_Name:" parsing artifact leaking into the value),
# and was substantially slower and less consistent (single-figure calls
# ranging from ~5s to ~230s on the same paper). Reverted; the doping-series
# coverage gap (structured_synthesis only has the general family formula,
# e.g. "CaFe1-xCoxAsF", not each plotted composition) remains open and
# needs a digitizer-PROMPT fix (resolve "x=0.05" + family name into the
# full numeric formula), not a model swap.
PLOT_DIGITIZATION_MODEL = "claude-sonnet-4.6"  # series-name digitization
LINKER_MODEL = "deepseek-v3.2"  # series name -> material linking

DEFAULT_PDF_DIR = "../../../data/pdf_papers_superconductors"
DEFAULT_OUTPUT_DIR = "../../../data/results_superconductors_hf"


def load_materials_by_paper_id() -> tuple[
    dict[str, list[str]], dict[str, dict[str, str]], dict[str, str]
]:
    """paper_id -> list of material names, paper_id -> {material:
    synthesis_method}, and paper_id -> paper text -- all read straight from
    the HF dataset (structured_synthesis, text_paper), no LLM call and no
    Mistral OCR needed for text (text_paper is populated for all 1384
    papers in this split)."""
    from datasets import load_dataset

    df = load_dataset(
        "parquet", data_files=DATASET_PATH, split="train"
    ).to_pandas()
    materials_out: dict[str, list[str]] = {}
    synthesis_method_out: dict[str, dict[str, str]] = {}
    paper_text_out: dict[str, str] = {}
    for _, row in df.iterrows():
        raw = row.get("structured_synthesis")
        if not raw:
            continue
        try:
            entries = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            continue
        names = [
            e.get("material_name") for e in entries if e.get("material_name")
        ]
        if not names:
            continue
        # sanitize to match download_pdfs.py's filename convention
        # (old-style ids like "cond-mat/0102313" have a "/")
        paper_id = row["id"].replace("/", "_")
        materials_out[paper_id] = names
        synthesis_method_out[paper_id] = {
            e["material_name"]: e["recipe"]["synthesis_method"]
            for e in entries
            if e.get("material_name")
            and (e.get("recipe") or {}).get("synthesis_method")
        }
        paper_text_out[paper_id] = row.get("text_paper") or ""
    return materials_out, synthesis_method_out, paper_text_out


def build_pipeline():
    """Figure extraction + plot digitization + series linking.

    materials/synthesis_method come from structured_synthesis (HF dataset),
    so material_extractor/synthesis_extractor stay unused and
    process_paper() is never called -- process_one() manually orchestrates
    extract_figures -> extract_plot_data -> plot_filter -> series_linker ->
    TcVLMProcessor.process(), same stages SynthesisPerformancePipeline uses
    internally, same convention as the catalysis/BatchRunner wiring
    (src/llm_synthesis/runners/batch_runner.py _init_components).
    """
    from llm_synthesis.config.plot_filter_config import PlotFilterConfig
    from llm_synthesis.services.pipelines.synthesis_performance_pipeline import (  # noqa: E501
        SynthesisPerformancePipeline,
    )
    from llm_synthesis.transformers.performance_linking.series_material_linker import (  # noqa: E501
        SeriesMaterialLinker,
    )
    from llm_synthesis.transformers.plot_extraction.litellm_plot_data_extraction import (  # noqa: E501
        LiteLLMPlotDataExtractor,
    )
    from llm_synthesis.utils.dspy_utils import get_llm_from_name
    from llm_synthesis.utils.llms import LLM_REGISTRY

    # Same resolution pattern as batch_runner.py _init_components: any
    # LLM_REGISTRY-registered VLM can digitize plots via
    # LiteLLMPlotDataExtractor
    # (same prompt/parsing logic as ClaudeLinePlotDataExtractor, routed through
    # litellm for multi-provider support) -- thermocatalysis's run.py benchmarks
    # exactly this across Gemini/Claude/Qwen for its digitizer choice.
    if PLOT_DIGITIZATION_MODEL in LLM_REGISTRY.configs:
        cfg_vlm = LLM_REGISTRY.configs[PLOT_DIGITIZATION_MODEL]
        plot_extractor = LiteLLMPlotDataExtractor(
            model=cfg_vlm.model,
            api_key=cfg_vlm.api_key,
            api_base=cfg_vlm.api_base,
            extra_kwargs=cfg_vlm.extra_kwargs or {},
        )
    else:
        plot_extractor = LiteLLMPlotDataExtractor(model=PLOT_DIGITIZATION_MODEL)
    linker_lm = get_llm_from_name(
        LINKER_MODEL, model_kwargs={"temperature": 0.0, "max_tokens": 32_000}
    )
    series_linker = SeriesMaterialLinker(lm=linker_lm)

    return SynthesisPerformancePipeline(
        material_extractor=None,
        synthesis_extractor=None,
        plot_extractor=plot_extractor,
        series_linker=series_linker,
        plot_filter_config=PlotFilterConfig.for_superconductivity(),
        figure_segmenter="florence",
    )


_FLORENCE_LOCK = threading.Lock()


def load_or_extract_figures(
    pipeline, pdf_extractor, pdf_path: Path, cache_dir: Path
) -> list:
    """Cache Florence-segmented figures per paper so re-running VLM
    experiments (different model/prompt) skips the ~9min CPU-bound OCR +
    Florence-2 segmentation step entirely.

    Paper text for digitization context comes from the HF dataset's own
    text_paper column (load_materials_by_paper_id), not from OCR -- no
    need to cache OCR'd text here.

    Florence-2 is a single shared CPU model instance -- not thread-safe for
    concurrent forward passes, so figure extraction is serialized via
    _FLORENCE_LOCK even when --workers > 1.
    """
    from llm_synthesis.models.figure import FigureInfo

    paper_id = pdf_path.stem
    fig_cache_file = cache_dir / paper_id / "figures.json"
    if fig_cache_file.exists():
        raw = json.loads(fig_cache_file.read_text())
        return [FigureInfo(**f) for f in raw]

    markdown = pdf_extractor.forward(pdf_path.read_bytes())
    with _FLORENCE_LOCK:
        figures = pipeline.extract_figures(markdown)

    fig_cache_file.parent.mkdir(parents=True, exist_ok=True)
    fig_cache_file.write_text(
        json.dumps([f.model_dump() for f in figures], indent=2)
    )
    return figures


def load_or_digitize_plots(
    pipeline, paper_id: str, figures: list, paper_text: str, cache_dir: Path
) -> tuple[list, list]:
    """Cache Claude digitization output (series names, axes) per paper so a
    later rerun (e.g. after a linking/Tc-prompt change that doesn't touch
    digitization) doesn't re-pay for the same Claude calls.

    Returns (plots, plot_figures) matching
    SynthesisPerformancePipeline.extract_plot_data's return shape.
    """
    from llm_synthesis.models.figure import FigureInfo
    from llm_synthesis.models.plot import ExtractedLinePlotData

    digitize_cache_file = cache_dir / paper_id / "digitized_plots.json"
    if digitize_cache_file.exists():
        raw = json.loads(digitize_cache_file.read_text())
        plots = [ExtractedLinePlotData(**d["plot"]) for d in raw]
        plot_figures = [FigureInfo(**d["figure"]) for d in raw]
        return plots, plot_figures

    plots, plot_figures = pipeline.extract_plot_data(
        figures, paper_text=paper_text, si_text=""
    )

    digitize_cache_file.parent.mkdir(parents=True, exist_ok=True)
    digitize_cache_file.write_text(
        json.dumps(
            [
                {"plot": p.model_dump(), "figure": f.model_dump()}
                for p, f in zip(plots, plot_figures)
            ],
            indent=2,
        )
    )
    return plots, plot_figures


class HFSynthesisMethodCsvWriter:
    """Wraps CsvMasterWriter, injecting synthesis_method from
    structured_synthesis (HF dataset) into each flat CSV record after the
    base writer builds it -- avoids needing a full GeneralSynthesisOntology
    object just to carry one string field through."""

    def __init__(self, synthesis_method_by_paper: dict[str, dict[str, str]]):
        from llm_synthesis.runners.output_writers.csv_writer import (
            CsvMasterWriter,
        )

        self._synthesis_method_by_paper = synthesis_method_by_paper
        self._writer = CsvMasterWriter(master_csv_name="tc_master.csv")

    def write_paper(self, *, paper_id, output_dir, pipeline_result, **kwargs):
        methods = self._synthesis_method_by_paper.get(paper_id, {})
        orig_build = self._writer._build_flat_records

        def _patched_build(pid, result, text_metrics, vlm_metrics):
            records = orig_build(pid, result, text_metrics, vlm_metrics)
            for rec in records:
                rec["synthesis_method"] = methods.get(rec["material"])
            return records

        self._writer._build_flat_records = _patched_build
        try:
            return self._writer.write_paper(
                paper_id=paper_id,
                output_dir=output_dir,
                pipeline_result=pipeline_result,
                **kwargs,
            )
        finally:
            self._writer._build_flat_records = orig_build

    def finalize(self, output_dir, all_summaries):
        return self._writer.finalize(output_dir, all_summaries)


def process_one(
    pipeline,
    pdf_extractor,
    pdf_path: Path,
    materials: list[str],
    cache_dir: Path,
    paper_text: str = "",
):
    from llm_synthesis.domain_metrics.superconductors.tc_vlm_processor import (
        TcVLMProcessor,
    )
    from llm_synthesis.services.pipelines.synthesis_performance_pipeline import (  # noqa: E501
        PipelineResult,
        SynthesisWithPerformanceEntry,
    )

    paper_id = pdf_path.stem
    logger.info(
        "Processing %s (%d materials from HF)", paper_id, len(materials)
    )

    figures = load_or_extract_figures(
        pipeline, pdf_extractor, pdf_path, cache_dir
    )

    # Digitize plots (series names, using full paper text for context) ->
    # filter to R(T)-relevant plots -> link each plot's series to a
    # material -> read Tc per plot, matching to materials via the linker's
    # plot_mappings (not a VLM self-match). Digitization (Claude, the
    # expensive step) is cached per paper -- see load_or_digitize_plots.
    plots, plot_figures = load_or_digitize_plots(
        pipeline, paper_id, figures, paper_text, cache_dir
    )
    relevant_plots, _ = pipeline.plot_filter.filter_plots(
        plots, log_skipped=False
    )
    plot_mappings = [
        mapping
        for idx, plot in relevant_plots
        if (
            mapping := pipeline._link_one_plot(
                idx, plot, plot_figures[idx], materials
            )
        )
        is not None
    ]

    tc_processor = TcVLMProcessor(claude_model=QWEN_MODEL)
    vlm_metrics = tc_processor.process(
        relevant_plots=relevant_plots,
        plot_figures=plot_figures,
        plot_mappings=plot_mappings,
        materials=materials,
        paper_text="",
    )

    # Fallback: the structured digitize->filter->link chain is more
    # accurate when every stage succeeds, but any single stage failing
    # (e.g. Florence-2 under-segmenting figures, or the digitizer
    # mislabeling axes so the plot filter drops a real R(T) plot) zeroes
    # out that material entirely -- a failure mode the old one-shot
    # VLM read (process_from_figures) doesn't have, since it free-
    # associates materials to curves straight from the raw image with no
    # intermediate stage to fail. Recover coverage for materials the
    # structured path missed, without discarding its results for the
    # materials it DID successfully link.
    missing = [m for m in materials if m not in vlm_metrics]
    if missing:
        fallback_metrics = tc_processor.process_from_figures(
            figures=figures, materials=missing
        )
        if fallback_metrics:
            logger.info(
                "  Fallback (one-shot VLM) recovered %d/%d missing "
                "materials: %s",
                len(fallback_metrics),
                len(missing),
                list(fallback_metrics),
            )
        for material, data in fallback_metrics.items():
            data["source"] = data.get("source", "main plot") + " (fallback)"
            vlm_metrics[material] = data

    # qwen_cost is genuinely per-paper (tc_processor is instantiated fresh
    # above, not shared across workers). digitizer/linker are single
    # instances shared across ALL worker threads (built once in
    # build_pipeline()) -- a per-paper before/after snapshot of their
    # cumulative cost is not attributable to this paper alone under
    # concurrent workers (one paper's "delta" picks up cost from whatever
    # other papers happen to be mid-digitization at the same time), so we
    # don't try to report a per-paper digitizer/linker split here. See
    # main()'s running total (pipeline.plot_extractor.get_cost() +
    # linker lm cost, read once at the end) for the real aggregate.
    qwen_cost = tc_processor.get_cost()
    logger.info("  Qwen (Tc-read) cost: $%.4f", qwen_cost)

    # DeepSeek tc_text extraction was tried and dropped based on the
    # 19-paper snippet run's Qwen-vs-human agreement -- that comparison
    # was later found to only cover n=10 matched rows (linking bug against
    # a stale ground-truth CSV); the honest number on n=27 correctly-linked
    # rows was R^2=0.54, MAE=7.6K, not R^2=0.97/MAE=1.5K. Re-evaluate
    # whether to bring text-Tc back once this linker-based VLM path's own
    # accuracy is re-measured against ground_truth_tc.xlsx.
    text_metrics: dict = {}

    results = [SynthesisWithPerformanceEntry(material=m) for m in materials]
    materials_with_perf = [m for m in materials if m in vlm_metrics]
    materials_without_perf = [m for m in materials if m not in vlm_metrics]

    return (
        PipelineResult(
            paper_id=paper_id,
            paper_name=paper_id,
            materials=materials,
            results=results,
            plot_mappings=plot_mappings,
            num_plots=len(figures),
            materials_with_performance=materials_with_perf,
            materials_without_performance=materials_without_perf,
            relevant_plots=relevant_plots,
            plot_figures=plot_figures,
        ),
        text_metrics,
        vlm_metrics,
        qwen_cost,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf_dir", nargs="?", default=DEFAULT_PDF_DIR)
    parser.add_argument("output_dir", nargs="?", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Papers to process concurrently (default: 4)",
    )
    parser.add_argument(
        "--shard",
        type=str,
        default=None,
        help=(
            "Run only 1/N of the queue, e.g. --shard 0/3. Lets multiple "
            "separate processes split the same PDF dir without racing on "
            "each other's _figure_cache writes or tc_master.csv appends "
            "(each shard still writes to output_dir/tc_master.csv, but on "
            "a disjoint set of papers -- no two shards touch the same "
            "paper_id, so no read-modify-write collision)."
        ),
    )
    parser.add_argument(
        "--paper-timeout",
        type=int,
        default=600,
        help=(
            "Seconds per paper before logging a 'SLOW' warning (default: "
            "600). Purely informational -- the paper's thread can't be "
            "canceled, so processing continues and the real result is "
            "still used once it finishes; this only flags unusually slow "
            "papers in the log."
        ),
    )
    parser.add_argument(
        "--max-cost",
        type=float,
        default=None,
        help=(
            "Stop submitting new papers once real spend (digitizer + "
            "linker + Tc-read, checked after each batch of --workers "
            "papers finishes) reaches this many dollars. Papers already "
            "in flight when the cap is hit still finish. Re-run with "
            "--skip-existing to continue past the cap later."
        ),
    )
    args = parser.parse_args()

    from llm_synthesis.transformers.pdf_extraction import MistralPDFExtractor

    pdf_dir = Path(args.pdf_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading materials from structured_synthesis (HF dataset)...")
    materials_by_id, synthesis_method_by_id, paper_text_by_id = (
        load_materials_by_paper_id()
    )
    logger.info(
        "  %d papers have materials in HF dataset", len(materials_by_id)
    )

    pdf_paths = sorted(pdf_dir.glob("*.pdf"))
    pdf_paths = [p for p in pdf_paths if p.stem in materials_by_id]
    if args.skip_existing:
        pdf_paths = [p for p in pdf_paths if not (output_dir / p.stem).exists()]
    # sorted() above is filename order = chronological (arxiv id) -- shuffle
    # so a partial/--max run doesn't just process the earliest years first.
    # Fixed seed -> every process sees the SAME shuffled order, which
    # --shard relies on to split disjoint ranges.
    random.Random(0).shuffle(pdf_paths)
    if args.shard:
        idx_str, n_str = args.shard.split("/")
        idx, n = int(idx_str), int(n_str)
        pdf_paths = pdf_paths[idx::n]
        logger.info("Shard %d/%d: %d papers", idx, n, len(pdf_paths))
    if args.max:
        pdf_paths = pdf_paths[: args.max]

    if not pdf_paths:
        logger.warning("No matching PDFs to process.")
        return

    pdf_extractor = MistralPDFExtractor(structured=False)
    pipeline = build_pipeline()
    writer = HFSynthesisMethodCsvWriter(synthesis_method_by_id)
    figure_cache_dir = output_dir / "_figure_cache"

    def _one(pdf_path: Path) -> dict:
        paper_start = time.time()
        try:
            materials = materials_by_id[pdf_path.stem]
            paper_text = paper_text_by_id.get(pdf_path.stem, "")
            result, text_metrics, vlm_metrics, cost = process_one(
                pipeline,
                pdf_extractor,
                pdf_path,
                materials,
                figure_cache_dir,
                paper_text=paper_text,
            )
            summary = writer.write_paper(
                paper_id=pdf_path.stem,
                output_dir=output_dir,
                pipeline_result=result,
                text_metrics=text_metrics,
                vlm_metrics=vlm_metrics,
                processing_time=time.time() - paper_start,
            )
            summary["vlm_cost_usd"] = cost
            logger.info("  Done: %s ($%.4f)", pdf_path.stem, cost)
            return summary
        except Exception as e:
            logger.error("FAILED %s: %s", pdf_path.stem, e)
            return {"paper_id": pdf_path.stem, "error": str(e)}

    def _running_cost(summaries: list[dict]) -> float:
        """Real aggregate cost so far: digitizer + linker are shared
        instances (read once, not per-paper -- see process_one for why a
        per-paper split of these two isn't attributable under concurrent
        workers), plus the sum of each paper's own Qwen cost."""
        digitizer_total = (
            pipeline.plot_extractor.get_cost()
            if hasattr(pipeline.plot_extractor, "get_cost")
            else 0.0
        )
        linker_total = sum(
            (h.get("cost") or 0.0)
            for h in getattr(pipeline.series_linker.lm, "history", [])
        )
        qwen_total = sum(s.get("vlm_cost_usd", 0) for s in summaries)
        return digitizer_total + linker_total + qwen_total

    total_start = time.time()
    # Paper-level parallelism: Florence-2 runs single-threaded per call
    # (CPU-bound, shared model instance), so most of the concurrency win
    # comes from overlapping Mistral OCR + Qwen VLM calls across papers.
    #
    # ponytail: --paper-timeout only WARNS past the budget, it does not
    # discard the future -- Python threads can't be force-stopped, so a
    # "timed out" call keeps running and eventually returns a real result
    # (measured: a call warned past 600s went on to finish at ~1167s with
    # a valid, billed result). Synthesizing a fake "timeout" summary at
    # the warning point would silently drop that real result AND its real
    # cost from the batch total. We still log the overrun so slow papers
    # are visible, but always wait for and use the actual return value.
    #
    # --max-cost is checked between batches of in-flight submissions (not
    # continuously) -- submit args.workers papers at a time, check the
    # real running total after each batch drains, stop submitting more
    # once over budget. Already-submitted papers in the batch still
    # finish and their real (paid-for) results are kept.
    summaries = []
    budget_exceeded = False
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for batch_start in range(0, len(pdf_paths), args.workers):
            if budget_exceeded:
                logger.warning(
                    "Stopping: $%.2f spent >= --max-cost $%.2f. "
                    "%d/%d papers not submitted (re-run with "
                    "--skip-existing to continue).",
                    _running_cost(summaries),
                    args.max_cost,
                    len(pdf_paths) - batch_start,
                    len(pdf_paths),
                )
                break

            batch = pdf_paths[batch_start : batch_start + args.workers]
            futures = {executor.submit(_one, p): p for p in batch}
            for future in futures:
                pdf_path = futures[future]
                try:
                    summaries.append(future.result(timeout=args.paper_timeout))
                except TimeoutError:
                    logger.warning(
                        "SLOW %s: exceeded %ds, still waiting for the "
                        "real result (thread can't be canceled)...",
                        pdf_path.stem,
                        args.paper_timeout,
                    )
                    summaries.append(future.result())

            if args.max_cost is not None:
                running = _running_cost(summaries)
                logger.info(
                    "  [budget] $%.2f spent so far (cap: $%.2f)",
                    running,
                    args.max_cost,
                )
                if running >= args.max_cost:
                    budget_exceeded = True

    writer.finalize(output_dir, summaries)
    total_cost = _running_cost(summaries)
    logger.info(
        "Batch done: %d/%d papers in %.1fs ($%.2f total cost: "
        "digitizer+linker+Tc-read). Results: %s",
        len(summaries),
        len(pdf_paths),
        time.time() - total_start,
        total_cost,
        output_dir,
    )


if __name__ == "__main__":
    main()
