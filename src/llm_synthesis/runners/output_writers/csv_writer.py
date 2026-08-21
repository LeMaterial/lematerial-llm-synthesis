"""JSON + growing flat CSV output writer.

Produces the same layout as the original superconductors batch_run_tc.py:
  <output_dir>/<paper_id>/
      <material>.json          — synthesis + evaluation + domain metrics
      summary.json             — per-paper counts
      tc_flat_records.jsonl    — flat records for this paper

  <output_dir>/
      <master_csv_name>        — growing CSV across all papers
                               (default: master.csv)
      batch_summary.json       — written by finalize()

The CSV columns are declared at construction time, making this writer
reusable for any domain that needs a flat master CSV.
"""

import csv
import json
import re
from pathlib import Path
from typing import Any

from llm_synthesis.runners.output_writers.base import BaseOutputWriter
from llm_synthesis.services.pipelines.synthesis_performance_pipeline import (
    PipelineResult,
)
from llm_synthesis.utils.formula_utils import normalize_formula
from llm_synthesis.utils.performance_utils import sanitize_filename

# Default CSV columns for the superconductors domain
SUPERCONDUCTOR_CSV_COLUMNS = [
    "paper_id",
    "year",
    "material",
    "material_normalized",
    "is_superconductor",
    "tc_text",
    "tc_text_onset",
    "tc_text_zero",
    "tc_vlm",
    "tc_vlm_onset",
    "tc_vlm_zero",
    "tc_vlm_source",
    "tc_vlm_source_plot",
    "tc_best",
    "tc_best_source",
    "has_text_tc",
    "has_vlm_tc",
    "synthesis_method",
    "synthesis_score",
]


def _extract_year_from_arxiv_id(paper_id: str) -> int | None:
    clean = paper_id.split("_")[0]
    clean = re.sub(r"v\d+$", "", clean)
    match = re.match(r"^(\d{2})(\d{2})\.\d+$", clean)
    if match:
        yy = int(match.group(1))
        return 2000 + yy if yy < 90 else 1900 + yy
    return None


def _pick_best_tc(
    text_tc: float | None,
    vlm_tc: float | None,
    text_onset: float | None,
) -> tuple[float | None, str]:
    if text_tc is not None:
        return text_tc, "text"
    if text_onset is not None:
        return text_onset, "text_onset"
    if vlm_tc is not None:
        return vlm_tc, "vlm"
    return None, "none"


def _append_to_master_csv(
    flat_records: list[dict],
    master_path: Path,
    columns: list[str],
) -> None:
    """Append records to master CSV, replacing existing rows for same paper."""
    master_path.parent.mkdir(parents=True, exist_ok=True)

    existing_keys: set[tuple[str, str]] = set()
    if master_path.exists() and master_path.stat().st_size > 0:
        with open(master_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_keys.add(
                    (row.get("paper_id", ""), row.get("material", ""))
                )

    new_keys = {(r["paper_id"], r["material"]) for r in flat_records}
    replace_keys = existing_keys & new_keys

    if replace_keys:
        all_rows: list[dict] = []
        if master_path.exists():
            with open(master_path, newline="") as f:
                reader = csv.DictReader(f)
                all_rows = [
                    row
                    for row in reader
                    if (
                        row.get("paper_id", ""),
                        row.get("material", ""),
                    )
                    not in replace_keys
                ]
        all_rows.extend(
            {k: (str(v) if v is not None else "") for k, v in r.items()}
            for r in flat_records
        )
        with open(master_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(all_rows)
    else:
        write_header = (
            not master_path.exists() or master_path.stat().st_size == 0
        )
        with open(master_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            if write_header:
                writer.writeheader()
            for rec in flat_records:
                writer.writerow(
                    {
                        k: (str(v) if v is not None else "")
                        for k, v in rec.items()
                    }
                )


class CsvMasterWriter(BaseOutputWriter):
    """Write per-paper JSON + summary, and append rows to a growing master CSV.

    Designed for domains that want a flat tabular dataset across all papers
    (e.g. superconductors with one row per material per paper).

    Subclass and override ``_build_flat_records`` to produce domain-specific
    CSV rows from the pipeline result and domain metrics.

    Args:
        csv_columns: Ordered list of column names for the master CSV.
        master_csv_name: Filename of the master CSV (placed in output_dir).
        flat_records_filename: Filename for the per-paper JSONL of flat records.
    """

    def __init__(
        self,
        csv_columns: list[str] = SUPERCONDUCTOR_CSV_COLUMNS,
        master_csv_name: str = "master.csv",
        flat_records_filename: str = "flat_records.jsonl",
    ) -> None:
        self._columns = csv_columns
        self._master_csv_name = master_csv_name
        self._flat_records_filename = flat_records_filename
        self._all_flat_records: list[dict] = []

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

        flat_records = self._build_flat_records(
            paper_id, result, text_metrics, vlm_metrics
        )

        # Per-material JSON files
        for entry in result.results:
            mat_name = sanitize_filename(entry.material)
            text_m = {
                k: v
                for k, v in text_metrics.get(entry.material, {}).items()
                if not k.startswith("_")
            }
            vlm_m = vlm_metrics.get(entry.material, {})
            mat_data: dict[str, Any] = {
                "material": entry.material,
                "synthesis": (
                    entry.synthesis.model_dump() if entry.synthesis else None
                ),
                "evaluation": (
                    entry.evaluation.model_dump() if entry.evaluation else None
                ),
                "tc_from_text": text_m if text_m else None,
                "tc_from_vlm": vlm_m if vlm_m else None,
                "performance": (
                    entry.performance.model_dump()
                    if entry.performance
                    else None
                ),
            }
            with open(paper_dir / f"{mat_name}.json", "w") as f:
                json.dump(mat_data, f, indent=2, default=str)

        # Per-paper summary
        base = self._base_summary(result, processing_time)
        base["materials_with_text_metric"] = sum(
            1 for m in result.materials if text_metrics.get(m)
        )
        base["materials_with_vlm_metric"] = sum(
            1 for m in result.materials if vlm_metrics.get(m)
        )
        with open(paper_dir / "summary.json", "w") as f:
            json.dump(base, f, indent=2, default=str)

        # Per-paper JSONL
        with open(paper_dir / self._flat_records_filename, "w") as f:
            for rec in flat_records:
                f.write(json.dumps(rec, default=str) + "\n")

        # Append to master CSV
        master_path = output_dir / self._master_csv_name
        _append_to_master_csv(flat_records, master_path, self._columns)

        self._all_flat_records.extend(flat_records)
        return base

    def finalize(self, output_dir: Path, all_summaries: list[dict]) -> None:
        super().finalize(output_dir, all_summaries)

    def _build_flat_records(
        self,
        paper_id: str,
        result: PipelineResult,
        text_metrics: dict[str, Any],
        vlm_metrics: dict[str, Any],
    ) -> list[dict]:
        """Build one flat CSV row per material.

        Override in a subclass for a fully custom CSV schema.
        The default implementation produces the superconductors schema.
        """
        year = _extract_year_from_arxiv_id(paper_id)
        records = []

        for entry in result.results:
            mat = entry.material
            text_entry = text_metrics.get(mat, {})
            vlm_entry = vlm_metrics.get(mat, {})

            text_tc = text_entry.get("Tc_mid")
            text_onset = text_entry.get("T_onset")
            text_zero = text_entry.get("T_zero")
            text_sc = text_entry.get("superconducting")

            vlm_tc = vlm_entry.get("Tc_mid")
            vlm_onset = vlm_entry.get("T_onset")
            vlm_zero = vlm_entry.get("T_zero")
            vlm_sc = vlm_entry.get("superconducting")
            vlm_source = (
                vlm_entry.get("source", "main plot") if vlm_entry else None
            )

            vlm_source_plot = None
            for mapping in result.plot_mappings:
                for sm in mapping.mappings:
                    if sm.material_name == mat:
                        vlm_source_plot = mapping.figure_reference
                        break
                if vlm_source_plot:
                    break

            if text_sc is not None:
                is_sc = text_sc
            elif vlm_sc is not None:
                is_sc = vlm_sc
            else:
                is_sc = None

            tc_best, tc_best_source = _pick_best_tc(text_tc, vlm_tc, text_onset)

            synth_method = (
                entry.synthesis.synthesis_method if entry.synthesis else None
            )
            synth_score = (
                entry.evaluation.scores.overall_score
                if entry.evaluation and entry.evaluation.scores
                else None
            )

            records.append(
                {
                    "paper_id": paper_id,
                    "year": year,
                    "material": mat,
                    "material_normalized": normalize_formula(mat),
                    "is_superconductor": is_sc,
                    "tc_text": text_tc,
                    "tc_text_onset": text_onset,
                    "tc_text_zero": text_zero,
                    "tc_vlm": vlm_tc,
                    "tc_vlm_onset": vlm_onset,
                    "tc_vlm_zero": vlm_zero,
                    "tc_vlm_source": vlm_source,
                    "tc_vlm_source_plot": vlm_source_plot,
                    "tc_best": tc_best,
                    "tc_best_source": tc_best_source,
                    "has_text_tc": text_tc is not None,
                    "has_vlm_tc": vlm_tc is not None,
                    "synthesis_method": synth_method,
                    "synthesis_score": synth_score,
                }
            )

        return records
