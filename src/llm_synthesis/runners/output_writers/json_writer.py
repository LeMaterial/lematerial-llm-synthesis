"""JSON output writer with human-annotation template.

Produces the same file layout as the original thermocatalysis run_all_papers.py:
  <output_dir>/<paper_id>/
      <material>.json              — synthesis + evaluation + performance
      performance_mappings.json   — raw series-to-material mappings
      linking_summary_llm.json    — summary + LLM linking evaluation
      linking_summary_human.json  — summary + blank fields for human annotation
"""

import json
from pathlib import Path
from typing import Any

from llm_synthesis.runners.output_writers.base import BaseOutputWriter
from llm_synthesis.services.pipelines.synthesis_performance_pipeline import (
    PipelineResult,
)
from llm_synthesis.utils.performance_utils import sanitize_filename

_EMPTY_LINKING_EVALUATION = {
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


class AnnotatedJsonWriter(BaseOutputWriter):
    """Write per-paper JSON files plus LLM and human annotation summaries.

    Optionally merges domain-specific text/VLM metrics into each material
    file under a configurable key (default: ``"domain_metrics"``).

    Args:
        domain_metrics_key: Key under which to store the merged
            text_metrics + vlm_metrics in each material JSON.
            Pass None to omit domain metrics from material files.
    """

    def __init__(
        self, domain_metrics_key: str | None = "domain_metrics"
    ) -> None:
        self._metrics_key = domain_metrics_key

    def write_paper(
        self,
        paper_id: str,
        output_dir: Path,
        pipeline_result: PipelineResult,
        text_metrics: dict[str, Any],
        vlm_metrics: dict[str, Any],
        processing_time: float,
    ) -> dict:
        paper_dir = self._paper_dir(output_dir, paper_id)
        result = pipeline_result

        # Per-material files
        for entry in result.results:
            mat_name = sanitize_filename(entry.material)
            mat_data: dict[str, Any] = {
                "material": entry.material,
                "synthesis": (
                    entry.synthesis.model_dump() if entry.synthesis else None
                ),
                "evaluation": (
                    entry.evaluation.model_dump() if entry.evaluation else None
                ),
                "performance": (
                    entry.performance.model_dump()
                    if entry.performance
                    else None
                ),
            }
            if self._metrics_key is not None:
                merged: dict[str, Any] = {}
                merged.update(text_metrics.get(entry.material, {}))
                merged.update(vlm_metrics.get(entry.material, {}))
                if merged:
                    mat_data[self._metrics_key] = merged

            with open(paper_dir / f"{mat_name}.json", "w") as f:
                json.dump(mat_data, f, indent=2, default=str)

        # Performance mappings
        if result.plot_mappings:
            with open(paper_dir / "performance_mappings.json", "w") as f:
                json.dump(
                    [m.model_dump() for m in result.plot_mappings],
                    f,
                    indent=2,
                )

        base = self._base_summary(result, processing_time)

        # LLM linking evaluation
        linking_evaluation = None
        if result.results and result.results[0].linking_evaluation:
            linking_evaluation = result.results[0].linking_evaluation

        llm_summary = {**base}
        llm_summary["linking_evaluation"] = (
            linking_evaluation.model_dump() if linking_evaluation else None
        )
        with open(paper_dir / "linking_summary_llm.json", "w") as f:
            json.dump(llm_summary, f, indent=2, default=str)

        human_summary = {**base}
        human_summary["linking_evaluation"] = _EMPTY_LINKING_EVALUATION
        with open(paper_dir / "linking_summary_human.json", "w") as f:
            json.dump(human_summary, f, indent=2, default=str)

        return base
