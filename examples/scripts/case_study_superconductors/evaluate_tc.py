"""Evaluate VLM Tc extraction against human-annotated ground truth.

Usage:
    python evaluate_tc.py [--results PATH_TO_CSV]

If --results is omitted, defaults to the known tc_master_snippet.csv path.
Loads ground truth from ground_truth_tc.xlsx (same directory as this script).

Outputs:
  - Three tables: Matched, Missing (GT only), Extra (VLM only)
  - Aggregate metrics: MAE, RMSE, recall, FP count
  - Full comparison CSV saved to evaluate_tc_output.csv
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent


def normalize_formula(s: str) -> str:
    """Normalize a chemical formula for fuzzy matching.

    Copied from src/llm_synthesis/utils/formula_utils.py to avoid heavy
    transitive imports (peft, torch, etc.).
    """
    base = s.strip()
    base = re.sub(r"\\(?:mathrm|text|textit|mathit|mathbf)\{([^}]*)\}", r"\1", base)
    base = re.sub(r"[_^]\{([^}]*)\}", r"\1", base)
    base = base.replace("$", "").replace("\\", "")
    base = re.sub(r"\s*\([^)]*\)\s*$", "", base).strip()
    base = re.sub(r"\s*\[[^\]]*\]\s*$", "", base).strip()
    base = base.replace("\u03b4", "delta").replace("\u0394", "delta")
    for char, digit in [
        ("\u2080", "0"), ("\u2081", "1"), ("\u2082", "2"), ("\u2083", "3"),
        ("\u2084", "4"), ("\u2085", "5"), ("\u2086", "6"), ("\u2087", "7"),
        ("\u2088", "8"), ("\u2089", "9"), ("\u208b", "-"),
    ]:
        base = base.replace(char, digit)
    base = base.lower().replace(" ", "").replace("\u2212", "-")
    base = re.sub(r"(\.\d*?)0+(?=\D|$)", r"\1", base)
    return base

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_GT_PATH = _SCRIPT_DIR / "ground_truth_tc.xlsx"
DEFAULT_RESULTS_PATH = Path(
    "/Users/valeriegentzke/Documents/Studieren/lemat_synth/"
    "lematerial-llm-synthesis/examples/data/pdf_papers/"
    "superconductor_pdfs-random-sample/selected_papers/"
    "results_Tc_v2_20_02/tc_master_snippet.csv"
)


def load_ground_truth(path: Path) -> pd.DataFrame:
    """Load and clean ground truth Excel file."""
    gt = pd.read_excel(path)
    # Ensure expected columns
    required = ["paper_id", "material", "tc_human_from_plot"]
    for col in required:
        if col not in gt.columns:
            raise ValueError(f"Ground truth missing column: {col}")
    gt["paper_id"] = gt["paper_id"].astype(str).str.strip()
    gt["material"] = gt["material"].astype(str).str.strip()
    gt["material_norm"] = gt["material"].apply(normalize_formula)
    # Parse tc_human_from_plot as numeric (may have NaN)
    gt["tc_human"] = pd.to_numeric(gt["tc_human_from_plot"], errors="coerce")
    return gt


def load_vlm_results(path: Path) -> pd.DataFrame:
    """Load VLM results CSV."""
    vlm = pd.read_csv(path)
    vlm["paper_id"] = vlm["paper_id"].astype(str).str.strip()
    vlm["material"] = vlm["material"].astype(str).str.strip()
    vlm["material_norm"] = vlm["material"].apply(normalize_formula)
    # Use tc_vlm_orig as the primary VLM Tc column
    if "tc_vlm_orig" in vlm.columns:
        vlm["tc_vlm"] = pd.to_numeric(vlm["tc_vlm_orig"], errors="coerce")
    elif "tc_best" in vlm.columns:
        vlm["tc_vlm"] = pd.to_numeric(vlm["tc_best"], errors="coerce")
    else:
        vlm["tc_vlm"] = np.nan
    return vlm


def match_gt_vlm(gt: pd.DataFrame, vlm: pd.DataFrame) -> pd.DataFrame:
    """Full outer join on (paper_id, material_norm).

    Returns DataFrame with columns:
      paper_id, gt_material, vlm_material, tc_human, tc_vlm, category, error
    """
    # Deduplicate VLM results per (paper_id, material_norm) — keep first non-NaN tc_vlm
    vlm_sorted = vlm.sort_values("tc_vlm", ascending=False, na_position="last")
    vlm_dedup = vlm_sorted.drop_duplicates(
        subset=["paper_id", "material_norm"], keep="first"
    )

    # Full outer join
    merged = pd.merge(
        gt[["paper_id", "material", "material_norm", "tc_human"]],
        vlm_dedup[["paper_id", "material", "material_norm", "tc_vlm"]],
        on=["paper_id", "material_norm"],
        how="outer",
        suffixes=("_gt", "_vlm"),
    )

    # Determine category
    has_gt = merged["material_gt"].notna()
    has_vlm = merged["material_vlm"].notna()
    merged["category"] = "extra"  # default: VLM only
    merged.loc[has_gt & has_vlm, "category"] = "matched"
    merged.loc[has_gt & ~has_vlm, "category"] = "missing"

    # Rename for clarity
    merged = merged.rename(
        columns={"material_gt": "gt_material", "material_vlm": "vlm_material"}
    )

    # Compute error for matched rows with numeric values
    matched_mask = (merged["category"] == "matched") & merged["tc_human"].notna() & merged["tc_vlm"].notna()
    merged["error"] = np.nan
    merged.loc[matched_mask, "error"] = (
        merged.loc[matched_mask, "tc_vlm"] - merged.loc[matched_mask, "tc_human"]
    )

    return merged


def print_report(comparison: pd.DataFrame) -> None:
    """Print the three tables and aggregate metrics."""
    matched = comparison[comparison["category"] == "matched"].copy()
    missing = comparison[comparison["category"] == "missing"].copy()
    extra = comparison[comparison["category"] == "extra"].copy()

    # --- Matched ---
    print("\n" + "=" * 80)
    print("MATCHED (Ground Truth ∩ VLM)")
    print("=" * 80)
    if len(matched) > 0:
        display_cols = [
            "paper_id", "gt_material", "vlm_material",
            "tc_human", "tc_vlm", "error",
        ]
        print(matched[display_cols].to_string(index=False))
    else:
        print("  (none)")

    # --- Missing ---
    print("\n" + "=" * 80)
    print("MISSING (Ground Truth only — VLM missed these)")
    print("=" * 80)
    if len(missing) > 0:
        display_cols = ["paper_id", "gt_material", "tc_human"]
        print(missing[display_cols].to_string(index=False))
    else:
        print("  (none)")

    # --- Extra ---
    print("\n" + "=" * 80)
    print("EXTRA (VLM only — false positives / duplicates)")
    print("=" * 80)
    if len(extra) > 0:
        display_cols = ["paper_id", "vlm_material", "tc_vlm"]
        print(extra[display_cols].to_string(index=False))
    else:
        print("  (none)")

    # --- Aggregate metrics ---
    print("\n" + "=" * 80)
    print("AGGREGATE METRICS")
    print("=" * 80)

    n_gt = len(comparison[comparison["category"].isin(["matched", "missing"])])
    n_matched = len(matched)
    n_missing = len(missing)
    n_extra = len(extra)

    print(f"  Ground truth rows:  {n_gt}")
    print(f"  Matched:            {n_matched}")
    print(f"  Missing (FN):       {n_missing}")
    print(f"  Extra (FP):         {n_extra}")
    print(f"  Recall:             {n_matched / n_gt:.1%}" if n_gt > 0 else "  Recall: N/A")

    # MAE / RMSE on matched rows with both numeric values
    errors = matched["error"].dropna()
    if len(errors) > 0:
        mae = errors.abs().mean()
        rmse = np.sqrt((errors ** 2).mean())
        print(f"  MAE (matched):      {mae:.2f} K")
        print(f"  RMSE (matched):     {rmse:.2f} K")
        print(f"  N with numeric Tc:  {len(errors)}")
    else:
        print("  MAE / RMSE:         N/A (no matched rows with numeric Tc)")


def main():
    parser = argparse.ArgumentParser(description="Evaluate VLM Tc extraction")
    parser.add_argument(
        "--results",
        type=Path,
        default=DEFAULT_RESULTS_PATH,
        help="Path to VLM results CSV (tc_master_snippet.csv)",
    )
    parser.add_argument(
        "--gt",
        type=Path,
        default=DEFAULT_GT_PATH,
        help="Path to ground truth Excel file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_SCRIPT_DIR / "evaluate_tc_output.csv",
        help="Path to save full comparison CSV",
    )
    args = parser.parse_args()

    print(f"Ground truth: {args.gt}")
    print(f"VLM results:  {args.results}")

    gt = load_ground_truth(args.gt)
    vlm = load_vlm_results(args.results)

    comparison = match_gt_vlm(gt, vlm)

    print_report(comparison)

    # Save full comparison
    comparison.to_csv(args.output, index=False)
    print(f"\nFull comparison saved to: {args.output}")


if __name__ == "__main__":
    main()
