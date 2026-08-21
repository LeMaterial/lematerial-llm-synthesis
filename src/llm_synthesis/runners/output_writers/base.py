"""Abstract base class for batch-runner output writers."""

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from llm_synthesis.services.pipelines.synthesis_performance_pipeline import (
    PipelineResult,
)


class BaseOutputWriter(ABC):
    """Write per-paper results and an end-of-batch summary.

    Subclasses implement write_paper() to produce whatever file format the
    domain requires.  The default finalize() writes a batch_summary.json;
    override it if you need additional aggregation (e.g. a master CSV).
    """

    @abstractmethod
    def write_paper(
        self,
        paper_id: str,
        output_dir: Path,
        pipeline_result: PipelineResult,
        text_metrics: dict[str, Any],
        vlm_metrics: dict[str, Any],
        processing_time: float,
    ) -> dict:
        """Persist all results for one paper and return a summary dict.

        Args:
            paper_id: Identifier for the paper (usually file stem).
            output_dir: Root output directory.  Writer creates a sub-directory
                ``output_dir / paper_id`` if needed.
            pipeline_result: Full PipelineResult from
                SynthesisPerformancePipeline.
            text_metrics: Per-material metrics from BaseTextMetricExtractor
                (empty dict if no extractor configured).
            vlm_metrics: Per-material metrics from BaseVLMMetricProcessor
                (empty dict if no processor configured).
            processing_time: Wall-clock seconds taken to process this paper.

        Returns:
            A flat summary dict (paper_id, counts, timing) that will be
            collected by the batch runner and passed to finalize().
        """

    def finalize(
        self,
        output_dir: Path,
        all_summaries: list[dict],
    ) -> None:
        """Called once after all papers have been processed.

        Default implementation writes batch_summary.json.
        Override to add domain-specific aggregation (e.g. master CSV).
        """
        batch_summary = {
            "total_papers": len(all_summaries),
            "successful": sum(1 for s in all_summaries if "error" not in s),
            "failed": sum(1 for s in all_summaries if "error" in s),
            "papers": all_summaries,
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "batch_summary.json", "w") as f:
            json.dump(batch_summary, f, indent=2, default=str)

    # ------------------------------------------------------------------
    # Shared helpers available to all writers
    # ------------------------------------------------------------------

    @staticmethod
    def _paper_dir(output_dir: Path, paper_id: str) -> Path:
        d = output_dir / paper_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    @staticmethod
    def _base_summary(
        pipeline_result: PipelineResult,
        processing_time: float,
    ) -> dict:
        """Build the subset of summary fields that every writer shares."""
        result = pipeline_result
        summary: dict[str, Any] = {
            "paper_id": result.paper_id,
            "paper_name": result.paper_name,
            "total_materials": len(result.materials),
            "materials_with_synthesis": sum(
                1 for r in result.results if r.synthesis
            ),
            "materials_with_performance": len(
                result.materials_with_performance
            ),
            "materials_without_performance": len(
                result.materials_without_performance
            ),
            "materials_list": result.materials,
            "materials_with_performance_list": (
                result.materials_with_performance
            ),
            "materials_without_performance_list": (
                result.materials_without_performance
            ),
            "total_plots_extracted": result.num_plots,
            "plots_linked": len(result.plot_mappings),
            "processing_time_seconds": round(processing_time, 1),
        }
        if result.linking_stats:
            stats = result.linking_stats
            summary["plots_skipped"] = {
                "not_relevant_x": stats.plots_skipped_not_relevant_x,
                "not_relevant_y": stats.plots_skipped_not_relevant_y,
                "no_series": stats.plots_skipped_no_series,
            }
            summary["confidence_breakdown"] = stats.confidence_counts
            summary["all_unmatched_series"] = stats.all_unmatched_series
        return summary
