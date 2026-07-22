#!/usr/bin/env python3
"""Superconductor case study, reusing text already in the HF dataset.

Skips material + synthesis extraction (both already in
`structured_synthesis` on LeMaterial/LeMat-Synth-Papers,
split=superconductor_keywords_and_LLM) and OCR-derived text metrics.
Only runs the steps that need fresh work per paper:

  1. OCR the PDF -> markdown w/ embedded figure images (Mistral, cached)
  2. Detect + segment figures (Florence-2)
  3. Read Tc via geometric construction AND match each curve to a
     material in one VLM call (Qwen via LiteLLM) -- no separate line-plot
     digitization or series->material linking pass.

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

QWEN_MODEL = "qwen3.5-397b-a17b"  # VLM: Tc reading + material matching

DEFAULT_PDF_DIR = "../../../data/pdf_papers_superconductors"
DEFAULT_OUTPUT_DIR = "../../../data/results_superconductors_hf"


def load_materials_by_paper_id() -> tuple[
    dict[str, list[str]], dict[str, dict[str, str]]
]:
    """paper_id -> list of material names, and paper_id -> {material:
    synthesis_method} -- both read straight from structured_synthesis
    (HF dataset), no LLM call."""
    from datasets import load_dataset

    df = load_dataset(
        "parquet", data_files=DATASET_PATH, split="train"
    ).to_pandas()
    materials_out: dict[str, list[str]] = {}
    synthesis_method_out: dict[str, dict[str, str]] = {}
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
    return materials_out, synthesis_method_out


def build_pipeline():
    """Figure extraction only -- no plot digitization, no series linker."""
    from llm_synthesis.config.plot_filter_config import PlotFilterConfig
    from llm_synthesis.services.pipelines.synthesis_performance_pipeline import (  # noqa: E501
        SynthesisPerformancePipeline,
    )

    # material_extractor / synthesis_extractor / plot_extractor /
    # series_linker are unused: we only call extract_figures(), then
    # TcVLMProcessor.process_from_figures() does Tc-reading + material
    # matching in a single VLM call. Never call process_paper().
    return SynthesisPerformancePipeline(
        material_extractor=None,
        synthesis_extractor=None,
        plot_filter_config=PlotFilterConfig.for_superconductivity(),
        figure_segmenter="florence",
    )


_FLORENCE_LOCK = threading.Lock()


def load_or_extract_figures(
    pipeline, pdf_extractor, pdf_path: Path, cache_dir: Path
):
    """Cache Florence-segmented figures per paper so re-running VLM
    experiments (different model/prompt) skips the ~9min CPU-bound OCR +
    Florence-2 segmentation step entirely.

    Mistral OCR is only needed for the embedded figure IMAGES (not in the
    HF dataset for this split) -- paper TEXT for tc_text extraction comes
    from the HF dataset's own text_paper column instead, so OCR'd markdown
    isn't cached/reused for text purposes.

    Florence-2 is a single shared CPU model instance -- not thread-safe for
    concurrent forward passes, so figure extraction is serialized via
    _FLORENCE_LOCK even when --workers > 1.
    """
    from llm_synthesis.models.figure import FigureInfo

    paper_id = pdf_path.stem
    cache_file = cache_dir / paper_id / "figures.json"
    if cache_file.exists():
        raw = json.loads(cache_file.read_text())
        return [FigureInfo(**f) for f in raw]

    markdown = pdf_extractor.forward(pdf_path.read_bytes())
    with _FLORENCE_LOCK:
        figures = pipeline.extract_figures(markdown)

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(
        json.dumps([f.model_dump() for f in figures], indent=2)
    )
    return figures


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

    tc_processor = TcVLMProcessor(claude_model=QWEN_MODEL)
    vlm_metrics = tc_processor.process_from_figures(
        figures=figures, materials=materials
    )
    logger.info("  VLM cost: $%.4f", tc_processor.get_cost())

    # DeepSeek tc_text extraction was tried and dropped: on the 19-paper
    # snippet run it agreed with Qwen's VLM readings far less (R^2=0.53,
    # MAE=9K) than Qwen agreed with human-curated ground truth (R^2=0.97,
    # MAE=1.5K) -- Qwen alone is the more trustworthy source here, and
    # tc_best previously preferred text over VLM whenever both existed,
    # which would have picked the noisier value.
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
            num_plots=len(figures),
            materials_with_performance=materials_with_perf,
            materials_without_performance=materials_without_perf,
        ),
        text_metrics,
        vlm_metrics,
        tc_processor.get_cost(),
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
    args = parser.parse_args()

    from llm_synthesis.transformers.pdf_extraction import MistralPDFExtractor

    pdf_dir = Path(args.pdf_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading materials from structured_synthesis (HF dataset)...")
    materials_by_id, synthesis_method_by_id = load_materials_by_paper_id()
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
            result, text_metrics, vlm_metrics, cost = process_one(
                pipeline,
                pdf_extractor,
                pdf_path,
                materials,
                figure_cache_dir,
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

    total_start = time.time()
    # Paper-level parallelism: Florence-2 runs single-threaded per call
    # (CPU-bound, shared model instance), so most of the concurrency win
    # comes from overlapping Mistral OCR + Qwen VLM calls across papers.
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        summaries = list(executor.map(_one, pdf_paths))

    writer.finalize(output_dir, summaries)
    total_cost = sum(s.get("vlm_cost_usd", 0) for s in summaries)
    logger.info(
        "Batch done: %d papers in %.1fs ($%.2f total VLM cost). Results: %s",
        len(pdf_paths),
        time.time() - total_start,
        total_cost,
        output_dir,
    )


if __name__ == "__main__":
    main()
