#!/usr/bin/env python3
"""
VLM plot extraction eval: compare LLM results to human ground truth.

Matches paper/material pairs across two directory trees, computes
RMSE and MAE per material using FigureExtractionMetric, and prints
a ranked summary table.

Usage:
    python eval_vlm.py \\
        --results  ../../../results/catalysis_llm \\
        --gt       ../../../data/results_catalysis_human \\
        [--metric  rmse|mae] \\
        [--csv     out.csv]

Directory conventions:
    results/  <paper_id>/<material>.json
        JSON must contain {"performance": {"plot_data":
        [{"series_name", "coordinates"}, ...]}}

    gt/       <paper_id>/<material>_human.json  (same performance schema)

Matching: paper_id must match exactly; material names are matched by
stripping _human suffix and normalizing (/ → -, % → pct, spaces → _).
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Allow running from any working directory
# ---------------------------------------------------------------------------
_repo_root = Path(__file__).resolve().parents[3]
if str(_repo_root / "src") not in sys.path:
    sys.path.insert(0, str(_repo_root / "src"))

from llm_synthesis.metrics.figure_extraction.figure_extraction_metric import (  # noqa: E402
    FigureExtractionMetric,
)
from llm_synthesis.models.plot import ExtractedLinePlotData  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_METRIC = FigureExtractionMetric()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalize(name: str) -> str:
    """Canonical material name for matching (lowercase, chars only)."""
    return (
        name.lower()
        .replace("/", "-")
        .replace("%", "pct")
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace(".", "")
    )


def _normalize_series(name: str) -> str:
    """Normalize a series name for fuzzy matching across VLM output formats.

    Handles:
    - Gemini LaTeX: $5\\% \\mathrm{La}/\\mathrm{Ni}/\\mathrm{Al}_2\\mathrm{O}_3$
    - Claude prefix: Series_Name (5%La/Ni/Al₂O₃)
    - Unicode subscripts: Al₂O₃ → Al2O3
    """
    import re

    # Strip "Series_Name:", "Series_Name (...)", "Series_Name ..." prefixes
    m = re.match(r"^Series_Name\s*[:(]?\s*(.*?)\)?$", name, re.IGNORECASE)
    if m:
        name = m.group(1).strip().rstrip(")")
    # Strip LaTeX math delimiters and commands
    name = re.sub(r"\$", "", name)
    name = re.sub(r"\\mathrm\{([^}]+)\}", r"\1", name)
    name = re.sub(r"\\text\{([^}]+)\}", r"\1", name)
    name = re.sub(r"\\[a-zA-Z]+", "", name)
    name = re.sub(r"\{|\}", "", name)
    # Normalize unicode subscript digits to ASCII
    _sub = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
    name = name.translate(_sub)
    # Lowercase, strip punctuation noise
    return (
        name.lower()
        .replace("%", "pct")
        .replace(" ", "")
        .replace("_", "")
        .replace("-", "")
        .replace("/", "")
        .replace("\\", "")
        .replace(".", "")
    )


def _load_performance(path: Path) -> dict[str, list[list[float]]] | None:
    """Return {series_name: [[x,y], ...]} from a result JSON, or None."""
    try:
        data = json.loads(path.read_text())
    except Exception as e:
        logger.warning("Cannot read %s: %s", path, e)
        return None

    perf = data.get("performance") or {}
    plot_data = perf.get("plot_data", [])
    if not plot_data:
        return None

    coords: dict[str, list[list[float]]] = {}
    for entry in plot_data:
        name = entry.get("series_name", "")
        c = entry.get("coordinates", [])
        if name and c:
            coords[_normalize_series(name)] = c
    return coords or None


def _to_extracted(
    coords: dict[str, list[list[float]]],
) -> ExtractedLinePlotData:
    return ExtractedLinePlotData(
        name_to_coordinates=coords,
        title=None,
        x_axis_label=None,
        x_axis_unit=None,
        y_left_axis_label=None,
        y_left_axis_unit=None,
    )


def _index_gt(gt_dir: Path) -> dict[str, dict[str, Path]]:
    """
    Returns {paper_id: {norm_material: path}} for every *_human.json in gt_dir.
    """
    index: dict[str, dict[str, Path]] = {}
    for paper_dir in sorted(gt_dir.iterdir()):
        if not paper_dir.is_dir():
            continue
        pid = paper_dir.name
        index[pid] = {}
        for f in paper_dir.glob("*_human.json"):
            mat_raw = f.stem.removesuffix("_human")
            index[pid][_normalize(mat_raw)] = f
    return index


def _index_results(results_dir: Path) -> dict[str, dict[str, Path]]:
    """
    Returns {paper_id: {norm_material: path}} for every *.json in results_dir,
    excluding summary/mapping files.
    """
    _skip = {
        "linking_summary_llm",
        "linking_summary_human",
        "performance_mappings",
        "batch_summary",
    }
    index: dict[str, dict[str, Path]] = {}
    for paper_dir in sorted(results_dir.iterdir()):
        if not paper_dir.is_dir():
            continue
        pid = paper_dir.name
        index[pid] = {}
        for f in paper_dir.glob("*.json"):
            if f.stem in _skip:
                continue
            index[pid][_normalize(f.stem)] = f
    return index


# ---------------------------------------------------------------------------
# Core eval
# ---------------------------------------------------------------------------


def evaluate(
    results_dir: Path,
    gt_dir: Path,
    metric: str = "rmse",
) -> list[dict]:
    """
    Match LLM results to GT and compute per-material error.

    Returns list of dicts:
      paper_id, material_gt, material_llm, score, n_gt_series,
      n_llm_series, n_matched_series
    """
    gt_index = _index_gt(gt_dir)
    res_index = _index_results(results_dir)

    rows = []
    matched_papers = set(gt_index) & set(res_index)
    unmatched_papers = set(gt_index) - set(res_index)

    if unmatched_papers:
        logger.warning(
            "GT papers with no LLM results (skipped): %s",
            sorted(unmatched_papers),
        )

    for pid in sorted(matched_papers):
        gt_mats = gt_index[pid]
        res_mats = res_index[pid]

        for norm_mat, gt_path in sorted(gt_mats.items()):
            # Find matching LLM result — exact norm match first
            if norm_mat in res_mats:
                res_path = res_mats[norm_mat]
            else:
                # Partial match fallback
                candidates = [
                    k for k in res_mats if norm_mat in k or k in norm_mat
                ]
                if not candidates:
                    logger.info(
                        "No LLM match for %s / %s (norm=%s)",
                        pid,
                        gt_path.stem,
                        norm_mat,
                    )
                    rows.append(
                        {
                            "paper_id": pid,
                            "material_gt": gt_path.stem.removesuffix("_human"),
                            "material_llm": None,
                            "score": None,
                            "n_gt_series": None,
                            "n_llm_series": None,
                            "n_matched_series": None,
                        }
                    )
                    continue
                res_path = res_mats[candidates[0]]
                logger.info(
                    "Fuzzy match: GT=%s → LLM=%s", norm_mat, res_path.stem
                )

            gt_coords = _load_performance(gt_path)
            llm_coords = _load_performance(res_path)

            mat_name_gt = gt_path.stem.removesuffix("_human")

            if not gt_coords:
                logger.info("No GT coords for %s / %s", pid, mat_name_gt)
                continue
            if not llm_coords:
                logger.info("No LLM coords for %s / %s", pid, res_path.stem)
                rows.append(
                    {
                        "paper_id": pid,
                        "material_gt": mat_name_gt,
                        "material_llm": res_path.stem,
                        "score": None,
                        "n_gt_series": len(gt_coords),
                        "n_llm_series": 0,
                        "n_matched_series": 0,
                    }
                )
                continue

            preds = _to_extracted(llm_coords)
            refs = _to_extracted(gt_coords)

            score = _METRIC(preds=preds, refs=refs, error_metric=metric)

            n_matched = len(set(llm_coords) & set(gt_coords))
            rows.append(
                {
                    "paper_id": pid,
                    "material_gt": mat_name_gt,
                    "material_llm": res_path.stem,
                    "score": score,
                    "n_gt_series": len(gt_coords),
                    "n_llm_series": len(llm_coords),
                    "n_matched_series": n_matched,
                }
            )

    return rows


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _print_table(rows: list[dict], metric: str) -> None:
    scored = [r for r in rows if r["score"] is not None]
    unscored = [r for r in rows if r["score"] is None]

    if not scored:
        print("No scoreable pairs found.")
        return

    avg = sum(r["score"] for r in scored) / len(scored)
    col_w = 36

    header = (
        f"{'Paper':<20}  {'Material (GT)':<{col_w}}  "
        f"{'Material (LLM)':<{col_w}}  {metric.upper():>8}  "
        f"{'GT ser':>6}  {'LLM ser':>7}  {'Matched':>7}"
    )
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))

    for r in sorted(scored, key=lambda x: (x["paper_id"], x["score"] or 0)):
        score_str = f"{r['score']:.4f}" if r["score"] is not None else "  N/A "
        print(
            f"{r['paper_id']:<20}  {r['material_gt']:<{col_w}}  "
            f"{(r['material_llm'] or ''):<{col_w}}  {score_str:>8}  "
            f"{r['n_gt_series']:>6}  {r['n_llm_series']:>7}  "
            f"{r['n_matched_series']:>7}"
        )

    print("-" * len(header))
    print(f"{'MEAN':>{20 + 2 + col_w + 2 + col_w + 2}} {avg:>8.4f}")
    print(f"\nScored: {len(scored)}  |  Missing/unmatched: {len(unscored)}")

    if unscored:
        print("\nUnscored materials:")
        for r in unscored:
            reason = (
                "no LLM match" if r["material_llm"] is None else "no coords"
            )
            print(f"  {r['paper_id']} / {r['material_gt']}  [{reason}]")


def _write_csv(rows: list[dict], path: Path, metric: str) -> None:
    import csv

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "paper_id",
                "material_gt",
                "material_llm",
                metric,
                "n_gt_series",
                "n_llm_series",
                "n_matched_series",
            ],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {
                    "paper_id": r["paper_id"],
                    "material_gt": r["material_gt"],
                    "material_llm": r["material_llm"],
                    metric: r["score"],
                    "n_gt_series": r["n_gt_series"],
                    "n_llm_series": r["n_llm_series"],
                    "n_matched_series": r["n_matched_series"],
                }
            )
    print(f"\nCSV saved to {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--results",
        required=True,
        help="Dir with LLM results: <paper_id>/<material>.json",
    )
    parser.add_argument(
        "--gt",
        default=str(_repo_root / "data" / "results_catalysis_human"),
        help="Ground truth dir (default: data/results_catalysis_human)",
    )
    parser.add_argument(
        "--metric",
        choices=["rmse", "mae"],
        default="rmse",
        help="Error metric (default: rmse)",
    )
    parser.add_argument("--csv", default=None, help="Optional CSV output path")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)

    results_dir = Path(args.results).resolve()
    gt_dir = Path(args.gt).resolve()

    if not results_dir.exists():
        print(f"ERROR: results dir not found: {results_dir}")
        sys.exit(1)
    if not gt_dir.exists():
        print(f"ERROR: ground truth dir not found: {gt_dir}")
        sys.exit(1)

    print(f"Results : {results_dir}")
    print(f"GT      : {gt_dir}")
    print(f"Metric  : {args.metric.upper()}")

    rows = evaluate(results_dir, gt_dir, metric=args.metric)
    _print_table(rows, args.metric)

    if args.csv:
        _write_csv(rows, Path(args.csv), args.metric)


if __name__ == "__main__":
    main()
