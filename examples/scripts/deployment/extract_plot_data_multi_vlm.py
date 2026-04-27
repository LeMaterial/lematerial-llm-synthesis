# This script is based on extract_synthesis_multi_llm_judge.py
# The logic from there is extended to run N VLMs on plot/figure extraction
# and generates a result.json summarizing the results per VLM.
#
# Usage - from papers — uses Florence-2 for figure classification followed by
# vlm extraction:
#   uv run examples/scripts/deployment/extract_plot_data_multi_vlm.py \
#     data_loader=default \
#     result_save=multi_llm
#
# Usage (from local PNGs — skips Florence, runs VLM directly on images):
#   uv run examples/scripts/deployment/extract_plot_data_multi_vlm.py \
#     from_plot_images=true
#
#   # Optionally specify a custom figure directory:
#   ... figure_dir=path/to/pngs
#
#   # Optionally change the ranking metric:
#   ... plot_extraction.rank_by=mean_rmse_norm

import asyncio
import base64
import json
import logging
import os
import random
import sys
import warnings
from typing import ClassVar

# Ensure the project root is on sys.path so sibling scripts are importable.
sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")),
)

import hydra
import numpy as np
from hydra.utils import get_original_cwd, instantiate
from omegaconf import DictConfig

from examples.scripts.evaluation.eval_utils import (
    compare_extraction_to_gt,
    parse_ground_truth_csv,
)
from llm_synthesis.data_loader.paper_loader.base import PaperLoaderInterface
from llm_synthesis.models.figure import FigureInfoWithPaper
from llm_synthesis.services.pipelines.plot_extraction_pipeline import (
    PlotExtractionPipeline,
)
from llm_synthesis.transformers.plot_extraction import LiteLLMPlotDataExtractor
from llm_synthesis.transformers.plot_extraction.claude_extraction import (
    resources,
)
from llm_synthesis.utils.concurrency import (
    get_max_concurrent_llm_calls,
    run_with_semaphore,
)
from llm_synthesis.utils.llms import LLM_REGISTRY

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
)
for _lg in ("pydantic", "litellm"):
    logging.getLogger(_lg).setLevel(logging.ERROR)


class VLMExtractorPool:
    """Reads VLM settings from Hydra config and builds extractors."""

    def __init__(self, cfg: DictConfig, prompt: str | None = None):
        """Initialise the pool from a Hydra ``plot_extraction`` config.

        Args:
            cfg: Full Hydra config (must contain ``plot_extraction``).
            prompt: Optional custom prompt forwarded to every extractor.
        """
        self.vlm_names = list(cfg.plot_extraction.vlm_names)
        self.max_tokens = int(cfg.plot_extraction.get("max_tokens", 1024))
        self.temperature = float(cfg.plot_extraction.get("temperature", 0.0))
        self.retry_temperatures = list(
            cfg.plot_extraction.get(
                "retry_temperatures", [self.temperature, 0.3, 0.5]
            )
        )
        self.extractors: dict[str, LiteLLMPlotDataExtractor] = {
            name: self._build(name, prompt) for name in self.vlm_names
        }

    def _build(
        self,
        vlm_name: str,
        prompt: str | None = None,
    ) -> LiteLLMPlotDataExtractor:
        """Build a single ``LiteLLMPlotDataExtractor`` for *vlm_name*."""
        if vlm_name in LLM_REGISTRY.configs:
            c = LLM_REGISTRY.configs[vlm_name]
            model, api_key, api_base = c.model, c.api_key, c.api_base
            extra_kwargs = c.extra_kwargs or {}
        else:
            model, api_key, api_base, extra_kwargs = vlm_name, None, None, {}

        kwargs = dict(
            model=model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            api_key=api_key,
            api_base=api_base,
            extra_kwargs=extra_kwargs,
            retry_temperatures=self.retry_temperatures,
        )
        if prompt is not None:
            kwargs["prompt"] = prompt
        return LiteLLMPlotDataExtractor(**kwargs)

    def __getitem__(self, vlm_name: str) -> LiteLLMPlotDataExtractor:
        """Return the extractor for *vlm_name*."""
        return self.extractors[vlm_name]


class MetricsEvaluator:
    """Compares VLM extractions to ground-truth CSVs and reports metrics."""

    def __init__(self, figure_dir: str, vlm_names: list[str]):
        """Initialise the evaluator.

        Args:
            figure_dir: Root directory containing figure subfolders.
            vlm_names: Ordered list of VLM identifiers to evaluate.
        """
        self.figure_dir = figure_dir
        self.vlm_names = vlm_names

    @staticmethod
    def _find_csv(figure_dir: str, subfolder: str) -> str | None:
        """Locate the ground-truth CSV for *subfolder*.

        Tries to find ``<subfolder>/<subfolder>.csv``; falls back to any
        ``.csv``in the subfolder.  Returns ``None`` when no CSV is found.
        """
        sub_path = os.path.join(figure_dir, subfolder)
        exact = os.path.join(sub_path, f"{subfolder}.csv")
        if os.path.isfile(exact):
            return exact
        for fname in os.listdir(sub_path):
            if fname.lower().endswith(".csv"):
                return os.path.join(sub_path, fname)
        return None

    def compute_metrics(self, all_results: dict[str, dict]):
        """Compare every (VLM, figure) extraction to its ground-truth CSV.

        Args:
            all_results:
            ``{subfolder: {"png": path, "vlms": {vlm: extraction}}}``.

        Returns:
            ``{subfolder: {vlm_name: metrics_dict, ...}, ...}``.
            Per-figure ``vlm_metrics.json`` files are saved alongside the PNGs.
        """
        all_metrics: dict[str, dict[str, dict]] = {}
        for subfolder, data in all_results.items():
            csv_path = self._find_csv(self.figure_dir, subfolder)
            if csv_path is None:
                logging.warning(f"  No CSV for {subfolder} — skipping metrics")
                continue
            gt_series = parse_ground_truth_csv(csv_path)
            if not gt_series:
                logging.warning(
                    f"  Empty CSV for {subfolder} — skipping metrics"
                )
                continue

            fig_metrics = {
                vlm: compare_extraction_to_gt(data["vlms"].get(vlm), gt_series)
                for vlm in self.vlm_names
            }
            all_metrics[subfolder] = fig_metrics
            self._save_json(
                fig_metrics,
                os.path.join(os.path.dirname(data["png"]), "vlm_metrics.json"),
            )
        return all_metrics

    # -- aggregate --

    def aggregate_metrics(
        self,
        all_metrics: dict[str, dict[str, dict]],
    ) -> dict[str, dict]:
        """Aggregate per-figure metrics into a single dict per VLM.

        Returns:
            ``{vlm_name: {num_successful, mean_rmse_norm, ...}, ...}``.
        """
        agg: dict[str, dict] = {}
        for vlm in self.vlm_names:
            ok = [
                all_metrics[s][vlm]
                for s in all_metrics
                if all_metrics[s][vlm].get("status") == "ok"
            ]
            rmses = [
                m["mean_rmse_norm"]
                for m in ok
                if m["mean_rmse_norm"] is not None
            ]
            maes = [
                m["mean_mae_norm"] for m in ok if m["mean_mae_norm"] is not None
            ]
            prs = [
                m["mean_pearson_r"]
                for m in ok
                if m.get("mean_pearson_r") is not None
            ]
            srs = [
                m["mean_spearman_rho"]
                for m in ok
                if m.get("mean_spearman_rho") is not None
            ]
            iccs = [m["mean_icc"] for m in ok if m.get("mean_icc") is not None]
            agg[vlm] = {
                "num_figures_with_gt": len(all_metrics),
                "num_successful": len(ok),
                "num_failed": len(all_metrics) - len(ok),
                "mean_rmse_norm": round(float(np.mean(rmses)), 4)
                if rmses
                else None,
                "mean_mae_norm": round(float(np.mean(maes)), 4)
                if maes
                else None,
                "mean_pearson_r": round(float(np.mean(prs)), 4)
                if prs
                else None,
                "mean_spearman_rho": round(float(np.mean(srs)), 4)
                if srs
                else None,
                "mean_icc": round(float(np.mean(iccs)), 4) if iccs else None,
                "total_series_matched": sum(
                    m["num_series_matched"] for m in ok
                ),
                "total_series_gt": sum(m["num_series_gt"] for m in ok),
            }
        return agg

    # -- ranking --

    RANK_METRICS: ClassVar[dict] = {
        "mean_rmse_norm": ("RMSE (norm)", True),  # lower is better
        "mean_mae_norm": ("MAE (norm)", True),  # lower is better
        "mean_pearson_r": ("Pearson r", False),  # higher is better
        "mean_spearman_rho": ("Spearman rho", False),  # higher is better
        "mean_icc": ("ICC", False),  # higher is better
    }

    def rank_vlms(
        self,
        aggregate: dict[str, dict],
        rank_by: str = "mean_pearson_r",
    ) -> list[dict]:
        """Rank VLMs by the chosen metric.

        Args:
            aggregate: Output of :meth:`aggregate_metrics`.
            rank_by: Key in ``RANK_METRICS`` to sort by.

        Returns:
            Sorted list of ``{rank, vlm, <all aggregate fields>}``.
        """
        if rank_by not in self.RANK_METRICS:
            logging.warning(
                f"Unknown rank_by='{rank_by}', falling back to 'mean_pearson_r'"
            )
            rank_by = "mean_pearson_r"

        _label, lower_is_better = self.RANK_METRICS[rank_by]
        entries = []
        for vlm, metrics in aggregate.items():
            val = metrics.get(rank_by)
            entries.append({"vlm": vlm, **metrics, "_sort_val": val})

        # Put None values at the end
        entries.sort(
            key=lambda e: (
                e["_sort_val"] is None,
                (e["_sort_val"] or 0)
                if lower_is_better
                else -(e["_sort_val"] or 0),
            )
        )
        for i, entry in enumerate(entries, 1):
            entry["rank"] = i
            del entry["_sort_val"]
        return entries

    # -- console table --

    def log_table(self, aggregate: dict[str, dict]) -> None:
        """Log the aggregate metrics as a console table."""
        w = 105
        logging.info("")
        logging.info("=" * w)
        logging.info("AGGREGATE METRICS vs GROUND TRUTH")
        logging.info("=" * w)
        logging.info(
            f"{'VLM':<25} {'OK':>4} {'Fail':>4} "
            f"{'RMSE':>8} {'MAE':>8} "
            f"{'Pearson':>8} {'Spearman':>9} {'ICC':>8} "
            f"{'Matched':>10}"
        )
        logging.info("-" * w)
        for vlm in self.vlm_names:
            a = aggregate[vlm]
            logging.info(
                f"{vlm:<25} {a['num_successful']:>4} {a['num_failed']:>4} "
                f"{(a['mean_rmse_norm'] or 0):>8.4f} "
                f"{(a['mean_mae_norm'] or 0):>8.4f} "
                f"{(a.get('mean_pearson_r') or 0):>8.4f} "
                f"{(a.get('mean_spearman_rho') or 0):>9.4f} "
                f"{(a.get('mean_icc') or 0):>8.4f} "
                f"{a['total_series_matched']:>4}/{a['total_series_gt']}"
            )
        logging.info("=" * w)

    def log_ranking_table(
        self,
        ranked: list[dict],
        rank_by: str = "mean_pearson_r",
    ) -> None:
        """Log VLM ranking as a formatted table."""
        label, _lower = self.RANK_METRICS.get(rank_by, (rank_by, False))
        w = 105
        logging.info("")
        logging.info("=" * w)
        logging.info("VLM RANKING (by %s)", label)
        logging.info("=" * w)
        logging.info(
            f"{'Rank':>4}  {'VLM':<25} {label:>12} "
            f"{'RMSE':>8} {'MAE':>8} {'Pearson':>8}"
            f" {'Spearman':>9} {'ICC':>8} {'OK':>4}"
        )
        logging.info("-" * w)
        for entry in ranked:
            val = entry.get(rank_by)
            logging.info(
                f"{entry['rank']:>4}  {entry['vlm']:<25} "
                f"{(val if val is not None else float('nan')):>12.4f} "
                f"{(entry.get('mean_rmse_norm') or 0):>8.4f} "
                f"{(entry.get('mean_mae_norm') or 0):>8.4f} "
                f"{(entry.get('mean_pearson_r') or 0):>8.4f} "
                f"{(entry.get('mean_spearman_rho') or 0):>9.4f} "
                f"{(entry.get('mean_icc') or 0):>8.4f} "
                f"{entry.get('num_successful', 0):>4}"
            )
        logging.info("=" * w)

    def _save_ranking_png(
        self,
        ranked: list[dict],
        rank_by: str = "mean_pearson_r",
    ) -> str:
        """Render a PNG table showing VLM ranking."""
        import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel

        label, _lower = self.RANK_METRICS.get(rank_by, (rank_by, False))
        headers = [
            "Rank",
            "VLM",
            label,
            "RMSE",
            "MAE",
            "Pearson",
            "Spearman",
            "ICC",
            "OK",
        ]
        table_data = []
        for entry in ranked:
            table_data.append(
                [
                    str(entry["rank"]),
                    entry["vlm"],
                    f"{entry.get(rank_by, 0) or 0:.4f}",
                    f"{entry.get('mean_rmse_norm', 0) or 0:.4f}",
                    f"{entry.get('mean_mae_norm', 0) or 0:.4f}",
                    f"{entry.get('mean_pearson_r', 0) or 0:.4f}",
                    f"{entry.get('mean_spearman_rho', 0) or 0:.4f}",
                    f"{entry.get('mean_icc', 0) or 0:.4f}",
                    str(entry.get("num_successful", 0)),
                ]
            )

        fig, ax = plt.subplots(
            figsize=(16, len(ranked) * 1.2 + 2),
        )
        ax.axis("off")
        tbl = ax.table(
            cellText=table_data,
            colLabels=headers,
            cellLoc="center",
            loc="center",
            bbox=[0, 0, 1, 1],
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(10)
        tbl.scale(1, 2.5)

        for col_idx in range(len(headers)):
            tbl[(0, col_idx)].set_facecolor("#4472C4")
            tbl[(0, col_idx)].set_text_props(weight="bold", color="white")
        for row_idx in range(1, len(table_data) + 1):
            for col_idx in range(len(headers)):
                if row_idx == 1:
                    tbl[(row_idx, col_idx)].set_facecolor("#D4EDDA")
                elif row_idx % 2 == 0:
                    tbl[(row_idx, col_idx)].set_facecolor("#E7E6E6")

        plt.title(
            f"VLM Ranking — Plot Data Extraction\nRanked by: {label}",
            fontsize=14,
            fontweight="bold",
            pad=20,
        )
        out_png = os.path.join(self.figure_dir, "vlm_ranking.png")
        plt.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()
        logging.info("Saved ranking PNG to %s", out_png)
        return out_png

    # -- convenience: full evaluate + save + log --

    def run(
        self,
        all_results: dict[str, dict],
        rank_by: str = "mean_pearson_r",
    ) -> None:
        """
        Full pipeline: compute per-figure metrics, aggregate, rank, 
        save JSONs, and log.
        """
        logging.info("Computing metrics against ground-truth CSVs …")
        all_metrics = self.compute_metrics(all_results)
        aggregate = self.aggregate_metrics(all_metrics)
        self._save_json(
            aggregate,
            os.path.join(self.figure_dir, "vlm_aggregate_metrics.json"),
        )
        self._save_json(
            all_metrics,
            os.path.join(self.figure_dir, "vlm_detailed_metrics.json"),
        )
        self.log_table(aggregate)

        # Rank VLMs and save ranking
        ranked = self.rank_vlms(aggregate, rank_by=rank_by)
        self.log_ranking_table(ranked, rank_by=rank_by)
        out_json = os.path.join(self.figure_dir, "vlm_ranking.json")
        self._save_json(
            [{"rank_by": rank_by, **e} for e in ranked],
            out_json,
        )
        self._save_ranking_png(ranked, rank_by=rank_by)
        logging.info("Saved VLM ranking to %s", out_json)

    @staticmethod
    def _save_json(data, path: str) -> None:
        """
        Write *data* as pretty-printed JSON to *path*, creating dirs as needed.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)


class BaseVLMRunner:
    """Base class providing shared VLM extraction logic.

    Subclasses implement ``run()`` to define how figures are gathered.
    The VLM execution is handled by ``_extract_figure`` and ``_batch_extract``.
    """

    def __init__(
        self, cfg: DictConfig, original_cwd: str, prompt: str | None = None
    ):
        """Initialise pool and concurrency settings.

        Args:
            cfg: Full Hydra config (must contain ``plot_extraction``).
            original_cwd: Original working directory (before Hydra changes it).
            prompt: Optional custom prompt forwarded to every extractor.
        """
        self.cfg = cfg
        self.original_cwd = original_cwd
        self.pool = VLMExtractorPool(cfg, prompt=prompt)
        self.max_concurrent = get_max_concurrent_llm_calls()

    async def _extract_figure(
        self,
        semaphore: asyncio.Semaphore,
        vlm_name: str,
        figure: FigureInfoWithPaper,
        label: str,
    ) -> tuple[str, str, dict | None, float]:
        """Run one VLM extractor on one figure.

        Args:
            semaphore: Shared concurrency limiter for LLM calls.
            vlm_name: VLM identifier.
            figure: The figure to extract data from.
            label: Human-readable label used for logging.

        Returns:
            ``(vlm_name, label, extraction_dict_or_None, cost)``.
        """
        extractor = self.pool[vlm_name]
        logging.info(f"  [{vlm_name}] -> {label}")
        try:
            result = await run_with_semaphore(
                semaphore, extractor.forward, figure
            )
            cost = extractor.get_cost()
            extractor.reset_cost()
            logging.info(
                f"    [{vlm_name}] {label}: "
                f"{len(result.name_to_coordinates)} series"
            )
            return vlm_name, label, result.model_dump(), cost
        except Exception as e:
            logging.error(f"    [{vlm_name}] {label} failed: {e}")
            return vlm_name, label, None, 0.0

    async def _batch_extract(
        self,
        semaphore: asyncio.Semaphore,
        labeled_figures: list[tuple[str, FigureInfoWithPaper]],
    ) -> tuple[list[tuple[str, str, dict | None, float]], float]:
        """Run all VLMs on all figures concurrently.

        Args:
            semaphore: Shared concurrency limiter.
            labeled_figures: ``[(label, figure), ...]``.

        Returns:
            ``(results, total_cost)`` where each result is
            ``(vlm_name, label, extraction_or_None, cost)``.
        """
        tasks = [
            self._extract_figure(semaphore, vlm, fig, label)
            for vlm in self.pool.vlm_names
            for label, fig in labeled_figures
        ]
        total_cost = 0.0
        results = []
        for item in await asyncio.gather(*tasks, return_exceptions=True):
            if isinstance(item, Exception):
                logging.error(f"Task failed: {item}")
                continue
            results.append(item)
            total_cost += item[3]
        return results, total_cost

    def run(self) -> None:
        """Execute the runner. Subclasses must implement."""
        raise NotImplementedError


class ImageBenchmarkRunner(BaseVLMRunner):
    """Extract plot data from local PNGs and benchmark against ground truth."""

    def __init__(self, cfg: DictConfig, original_cwd: str):
        """Initialise the runner.

        Args:
            cfg: Full Hydra config.
            original_cwd: Original working directory (before Hydra changes it).
        """
        super().__init__(cfg, original_cwd, prompt=resources.LINE_CHART_PROMPT)
        figure_dir = cfg.plot_extraction.get(
            "figure_dir", "examples/data/figure_labeling"
        )
        if not os.path.isabs(figure_dir):
            figure_dir = os.path.join(original_cwd, figure_dir)
        self.figure_dir = figure_dir

    @staticmethod
    def _png_to_base64(png_path: str) -> str:
        """Read a PNG file and return its base64-encoded string."""
        with open(png_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    @staticmethod
    def _collect_pngs(figure_dir: str) -> list[tuple[str, str]]:
        """Walk *figure_dir* and return ``[(subfolder, png_path), ...]``."""
        pngs = []
        for subfolder in sorted(os.listdir(figure_dir)):
            sub_path = os.path.join(figure_dir, subfolder)
            if not os.path.isdir(sub_path):
                continue
            for fname in os.listdir(sub_path):
                if fname.lower().endswith(".png"):
                    pngs.append((subfolder, os.path.join(sub_path, fname)))
        return pngs

    def _make_figure(self, png_path: str, name: str) -> FigureInfoWithPaper:
        """
        Wrap a local PNG into a minimal ``FigureInfoWithPaper`` 
        (no paper context).
        """
        return FigureInfoWithPaper(
            base64_data=self._png_to_base64(png_path),
            alt_text=name,
            position=0,
            context_before="",
            context_after="",
            figure_reference=name,
            figure_class="Graph plots",
            quantitative=True,
            paper_text="",
            si_text="",
        )

    def run(self) -> None:
        """
        Run VLM extraction on all PNGs, save results, and evaluate metrics.
        """
        pngs = self._collect_pngs(self.figure_dir)
        if not pngs:
            logging.error(f"No PNGs found in {self.figure_dir}")
            return

        vlm_names = self.pool.vlm_names
        logging.info(f"Found {len(pngs)} PNGs, VLMs: {vlm_names}")

        labeled_figures = [
            (sub, self._make_figure(path, sub)) for sub, path in pngs
        ]
        png_lookup = dict(pngs)

        logging.info(
            f"Processing {len(pngs)} × {len(vlm_names)} = "
            f"{len(pngs) * len(vlm_names)} tasks"
        )

        async def _run():
            semaphore = asyncio.Semaphore(self.max_concurrent)
            return await self._batch_extract(semaphore, labeled_figures)

        results, total_cost = asyncio.run(_run())

        # Organize results by subfolder
        all_results: dict[str, dict] = {}
        for vlm_name, subfolder, extraction, _cost in results:
            all_results.setdefault(
                subfolder,
                {"png": png_lookup[subfolder], "vlms": {}},
            )
            all_results[subfolder]["vlms"][vlm_name] = extraction

        # Save per-figure results
        for subfolder, data in all_results.items():
            out = os.path.join(os.path.dirname(data["png"]), "vlm_result.json")
            MetricsEvaluator._save_json(data["vlms"], out)
            logging.info(f"  Saved {out}")

        # Save extraction summary
        summary = {}
        for vlm in vlm_names:
            ok = [
                d["vlms"][vlm]
                for d in all_results.values()
                if d["vlms"].get(vlm) is not None
            ]
            n = sum(len(e.get("name_to_coordinates", {})) for e in ok)
            summary[vlm] = {
                "num_figures": len(pngs),
                "num_successful": len(ok),
                "total_series": n,
                "avg_series_per_figure": round(n / len(ok), 2) if ok else 0.0,
            }
        MetricsEvaluator._save_json(
            summary,
            os.path.join(self.figure_dir, "vlm_extraction_summary.json"),
        )
        logging.info(f"Total cost: ${total_cost:.6f}")

        # Evaluate against ground truth
        rank_by = self.cfg.plot_extraction.get("rank_by", "mean_pearson_r")
        evaluator = MetricsEvaluator(self.figure_dir, vlm_names)
        evaluator.run(all_results, rank_by=rank_by)
        logging.info("Success")


class PaperExtractionRunner(BaseVLMRunner):
    """Load papers, detect figures with Florence-2, run VLM extraction."""

    _pipeline = PlotExtractionPipeline()

    def __init__(self, cfg: DictConfig, original_cwd: str):
        """Initialise the runner.

        Args:
            cfg: Full Hydra config (must include ``data_loader``, 
            ``result_save``, and ``plot_extraction`` sections).
            original_cwd: Original working directory (before Hydra changes it).
        """
        super().__init__(cfg, original_cwd)

    def _resolve_paths(self) -> None:
        """Make relative data/annotation paths absolute using *original_cwd*."""
        arch = self.cfg.data_loader.architecture
        if hasattr(arch, "data_dir"):
            d = arch.data_dir
            if not (
                d.startswith("s3://")
                or d.startswith("gs://")
                or d.startswith("/")
            ):
                arch.data_dir = os.path.join(self.original_cwd, d)
        if hasattr(arch, "annotations_dir"):
            if not arch.annotations_dir.startswith("/"):
                arch.annotations_dir = os.path.join(
                    self.original_cwd, arch.annotations_dir
                )

    async def _get_text_and_figures(self, paper) -> tuple[str, list]:
        """
        Get paper text (downloading/converting the PDF if needed) and extract 
        figures.
        """
        paper_text = paper.publication_text
        has_images = "data:image/" in (paper_text or "")
        if paper.pdf_url and (
            not paper_text or len(paper_text.strip()) < 100 or not has_images
        ):
            logging.info(f"  Converting PDF from {paper.pdf_url}...")
            try:
                paper_text = await asyncio.to_thread(
                    self._pipeline.convert_pdf_from_url,
                    paper.pdf_url,
                )
                logging.info(f"  Converted: {len(paper_text)} chars")
            except Exception as e:
                logging.error(f"  PDF conversion failed: {e}")
                return paper_text, []
        figures = await asyncio.to_thread(
            self._pipeline.extract_figures, paper_text
        )
        return paper_text, figures

    def run(self) -> None:
        """Load papers, extract figures, run all VLMs, and save results."""
        self._resolve_paths()

        data_loader: PaperLoaderInterface = instantiate(
            self.cfg.data_loader.architecture,
        )
        papers = data_loader.load()
        if self.cfg.data_loader.number_of_samples:
            papers = random.sample(
                papers, self.cfg.data_loader.number_of_samples
            )

        vlm_names = self.pool.vlm_names
        logging.info(f"VLMs (n={len(vlm_names)}): {vlm_names}")

        result_gather = instantiate(self.cfg.result_save.architecture)
        result_dir = self.cfg.result_save.architecture.result_dir

        to_process = [p for p in papers if p.id not in os.listdir(result_dir)]
        if self.cfg.data_loader.number_of_samples:
            to_process = random.sample(
                to_process,
                self.cfg.data_loader.number_of_samples,
            )

        total_cost = 0.0
        llm_semaphore = asyncio.Semaphore(self.max_concurrent)
        max_concurrent_papers = 4
        paper_semaphore = asyncio.Semaphore(max_concurrent_papers)

        async def process_paper(paper) -> tuple:
            """
            Process a single paper: get figures, run VLMs via _batch_extract,
            save.
             """
            logging.info(f"Processing {paper.name}")
            try:
                paper_text, figures = await self._get_text_and_figures(paper)
                if not figures:
                    logging.warning(
                        f"  No quantitative figures in {paper.name}"
                    )
                    return None, 0.0

                # Build labeled figures for the shared _batch_extract
                fig_lookup: dict[str, object] = {}
                labeled_figures = []
                for fig in figures:
                    fig_wp = FigureInfoWithPaper(
                        **fig.model_dump(),
                        paper_text=paper_text,
                        si_text=paper.si_text,
                    )
                    labeled_figures.append((fig.figure_reference, fig_wp))
                    fig_lookup[fig.figure_reference] = fig

                results, paper_cost = await self._batch_extract(
                    llm_semaphore,
                    labeled_figures,
                )

                # Assemble results by VLM
                multi_vlm_results, cost_operations, summary = [], [], {}
                for vlm in vlm_names:
                    vlm_figs, vlm_cost = [], 0.0
                    for r_vlm, label, extraction, cost in results:
                        if r_vlm == vlm:
                            vlm_cost += cost
                            orig_fig = fig_lookup[label]
                            vlm_figs.append(
                                {
                                    "figure_reference": (
                                        orig_fig.figure_reference
                                    ),
                                    "figure_class": orig_fig.figure_class,
                                    "extraction": extraction,
                                }
                            )
                    multi_vlm_results.append({"vlm": vlm, "figures": vlm_figs})
                    cost_operations.append(
                        {
                            "operation": "plot_extraction",
                            "vlm": vlm,
                            "cost_usd": vlm_cost,
                        }
                    )
                    ok = [f for f in vlm_figs if f["extraction"]]
                    n_series = sum(
                        len(f["extraction"].get("name_to_coordinates", {}))
                        for f in ok
                    )
                    summary[vlm] = {
                        "avg_series_per_figure": (
                            round(n_series / len(ok), 2) if ok else 0.0
                        ),
                        "num_figures": len(figures),
                        "num_successful": len(ok),
                    }

                result_gather.gather(
                    paper_id=paper.id,
                    publication_text=paper_text,
                    si_text=paper.si_text,
                    multi_llm_results=multi_vlm_results,
                    cost_data=cost_operations,
                )
                paper_dir = os.path.join(result_dir, paper.id)
                MetricsEvaluator._save_json(
                    summary,
                    os.path.join(paper_dir, "vlm_summary.json"),
                )
                logging.info(f"  Done: {paper.name} (${paper_cost:.4f})")
                return summary, paper_cost
            except Exception as e:
                logging.error(f"Failed to process {paper.name}: {e}")
                return None, 0.0

        async def run_all():
            nonlocal total_cost
            if not to_process:
                return []

            async def run_one(paper):
                async with paper_semaphore:
                    return paper, await process_paper(paper)

            all_paper_results = []
            for item in await asyncio.gather(
                *[run_one(p) for p in to_process],
                return_exceptions=True,
            ):
                if isinstance(item, Exception):
                    logging.error(f"Paper task failed: {item}")
                    continue
                paper, (summ, cost) = item
                if summ is not None:
                    all_paper_results.append(summ)
                    total_cost += cost
                    logging.info(f"Finished {paper.name}: cost=${cost:.6f}")
            return all_paper_results

        logging.info(
            f"Processing {len(to_process)} papers "
            f"(max {max_concurrent_papers} papers, "
            f"{self.max_concurrent} LLM calls)"
        )
        all_paper_results = asyncio.run(run_all())

        # Global summary
        if all_paper_results:
            totals = {v: [] for v in vlm_names}
            for pr in all_paper_results:
                for v in vlm_names:
                    val = pr.get(v, {}).get("avg_series_per_figure")
                    if val is not None:
                        totals[v].append(val)
            global_summary = {
                v: {
                    "avg_series_per_figure": (
                        round(sum(vals) / len(vals), 2) if vals else 0.0
                    )
                }
                for v, vals in totals.items()
            }
            MetricsEvaluator._save_json(
                global_summary,
                os.path.join(result_dir, "global_avg_vlm_summary.json"),
            )

        logging.info(f"Total cost across all papers: ${total_cost:.6f}")
        logging.info("Success")


@hydra.main(
    config_path="../../config",
    config_name="config.yaml",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    """
    Hydra entry point — dispatches to ImageBenchmarkRunner or
    PaperExtractionRunner.
    """
    original_cwd = get_original_cwd()
    if cfg.plot_extraction.get("from_plot_images", False):
        ImageBenchmarkRunner(cfg, original_cwd).run()
    else:
        PaperExtractionRunner(cfg, original_cwd).run()


if __name__ == "__main__":
    main()
