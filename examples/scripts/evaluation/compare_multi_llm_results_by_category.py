"""Compare multi-LLM result.json with result_human.json and generate agreement 
statistics -- category wise breakdown."""

import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_multi_llm_results_complete import (  # pylint: disable=wrong-import-position
    load_annotations,
)
from eval_utils import (  # pylint: disable=wrong-import-position
    SCORE_COLUMNS,
    col_label,
    compute_agreement_metrics,
    merge_on_material_id,
)

OVERALL_COL = "overall_score"
OUTPUT_DIR = "results/agreement_analysis"

# inspired from compare_human_judge_scores_by_category.py
FULL_HEADER = (
    f"{'Label':<30} {'Rho':>6} {'p':>8} {'Kappa':>7} {'ICC2':>7} {'ICC3':>7} "
    f"{'H-Mean':>7} {'H-Med':>6} {'H-Std':>6} "
    f"{'L-Mean':>7} {'L-Med':>6} {'L-Std':>6} {'n':>4}"
)
COMPACT_HEADER = (
    f"{'Label':<30} {'Rho':>6} {'Kappa':>7} "
    f"{'H-Mean':>7} {'L-Mean':>7} {'AbsDiff':>7} {'n':>4}"
)


def _section(title):
    """Log a section divider."""
    logging.info("\n%s\n%s\n%s", "=" * 120, title, "=" * 120)


def _log_full_row(label, metrics):
    """Log one row with all agreement metrics."""
    logging.info(
        "%-30s %6.3f %8.4f %7.3f %7.3f %7.3f "
        "%7.2f %6.2f %6.2f %7.2f %6.2f %6.2f %4d",
        label,
        metrics["rho"],
        metrics["p"],
        metrics["kappa"],
        metrics["icc2"],
        metrics["icc3"],
        metrics["h_mean"],
        metrics["h_median"],
        metrics["h_std"],
        metrics["l_mean"],
        metrics["l_median"],
        metrics["l_std"],
        metrics["n"],
    )


def _log_compact_row(label, metrics):
    """Log one row with compact agreement metrics."""
    logging.info(
        "%-30s %6.3f %7.3f %7.2f %7.2f %7.3f %4d",
        label,
        metrics["rho"],
        metrics["kappa"],
        metrics["h_mean"],
        metrics["l_mean"],
        metrics["abs_diff"],
        metrics["n"],
    )


def analysis_by_score_category(human_df, llm_df, score_cols):
    """Log agreement metrics for each score column across all judges."""
    _section("AGREEMENT BY SCORE CATEGORY (all judges pooled)")
    merged = merge_on_material_id(human_df, llm_df, score_cols)
    logging.info("%s\n%s", FULL_HEADER, "-" * len(FULL_HEADER))
    for col in score_cols:
        metrics = compute_agreement_metrics(
            merged[f"{col}_h"], merged[f"{col}_l"]
        )
        if metrics:
            _log_full_row(col_label(col), metrics)


def analysis_by_group(human_df, llm_df, score_cols, group_col, group_label):
    """Log compact agreement for each unique value in *group_col*."""
    if OVERALL_COL not in score_cols:
        return
    _section(f"AGREEMENT BY {group_label} ({OVERALL_COL})")
    logging.info("%s\n%s", COMPACT_HEADER, "-" * len(COMPACT_HEADER))
    if group_col == "synth_llm":
        group_values = sorted(human_df[group_col].dropna().unique())
    else:
        group_values = sorted(llm_df["judge_id"].dropna().unique())

    for value in group_values:
        if group_col == "synth_llm":
            merged = merge_on_material_id(
                human_df[human_df["synth_llm"] == value],
                llm_df[llm_df["synth_llm"] == value],
                [OVERALL_COL],
            )
        else:
            merged = merge_on_material_id(
                human_df, llm_df[llm_df["judge_id"] == value], [OVERALL_COL]
            )
        metrics = compute_agreement_metrics(
            merged[f"{OVERALL_COL}_h"], merged[f"{OVERALL_COL}_l"]
        )
        if metrics:
            _log_compact_row(value, metrics)


def per_score_per_judge(human_df, llm_df, score_cols):
    """Log compact metrics for every (judge, score column) pair."""
    _section("PER JUDGE x PER SCORE CATEGORY BREAKDOWN")
    for judge in sorted(llm_df["judge_id"].dropna().unique()):
        merged = merge_on_material_id(
            human_df, llm_df[llm_df["judge_id"] == judge], score_cols
        )
        logging.info("\n--- Judge: %s ---", judge)
        logging.info("%s\n%s", COMPACT_HEADER, "-" * len(COMPACT_HEADER))
        for col in score_cols:
            metrics = compute_agreement_metrics(
                merged[f"{col}_h"], merged[f"{col}_l"]
            )
            if metrics:
                _log_compact_row(col_label(col), metrics)


def analyze_by_material_category(
    human_df, llm_df, score_cols, cat_col, cat_label
):
    """Per-judge agreement within each value of *cat_col*.

    Returns a list of dicts suitable for CSV export.
    """
    if cat_col not in human_df.columns or cat_col not in llm_df.columns:
        logging.info("Column %s not found in data, skipping.", cat_col)
        return []

    _section(f"ANALYSIS BY {cat_label} (per judge, {OVERALL_COL})")
    judge_names = sorted(llm_df["judge_id"].dropna().unique())
    cat_values = [c for c in sorted(human_df[cat_col].dropna().unique()) if c]
    logging.info("Found %d categories: %s", len(cat_values), cat_values)

    grid = {}  # cat_value -> judge -> metrics
    csv_rows = []

    for cat_value in cat_values:
        h_cat = human_df[human_df[cat_col] == cat_value]
        l_cat = llm_df[llm_df[cat_col] == cat_value]
        n_materials = h_cat["material_id"].nunique()
        if n_materials < 2:
            logging.info("\n  %s: skipped (n=%d)", cat_value, n_materials)
            continue

        logging.info("\n  Category: %s (%d materials)", cat_value, n_materials)
        judge_hdr = "    %-30s" + "".join(f" {j:>16s}" for j in judge_names)
        logging.info(judge_hdr, "Judge")

        abs_cells, rho_cells, kappa_cells = [], [], []
        best_judge, best_abs = None, float("inf")

        for judge in judge_names:
            merged = merge_on_material_id(
                h_cat, l_cat[l_cat["judge_id"] == judge], [OVERALL_COL]
            )
            metrics = compute_agreement_metrics(
                merged[f"{OVERALL_COL}_h"], merged[f"{OVERALL_COL}_l"]
            )
            if metrics:
                grid.setdefault(cat_value, {})[judge] = metrics
                abs_cells.append(f"{metrics['abs_diff']:>16.3f}")
                rho_cells.append(f"{metrics['rho']:>16.3f}")
                kappa_cells.append(f"{metrics['kappa']:>16.3f}")
                if metrics["abs_diff"] < best_abs:
                    best_abs, best_judge = metrics["abs_diff"], judge
                csv_rows.append(
                    {
                        "category_type": cat_label,
                        "category": cat_value,
                        "judge": judge,
                        "n": metrics["n"],
                        "rho": metrics["rho"],
                        "p": metrics["p"],
                        "kappa": metrics["kappa"],
                        "icc2": metrics["icc2"],
                        "icc3": metrics["icc3"],
                        "h_mean": metrics["h_mean"],
                        "h_median": metrics["h_median"],
                        "h_std": metrics["h_std"],
                        "l_mean": metrics["l_mean"],
                        "l_median": metrics["l_median"],
                        "l_std": metrics["l_std"],
                        "mean_diff": metrics["mean_diff"],
                        "abs_diff": metrics["abs_diff"],
                    }
                )
            else:
                abs_cells.append(f"{'N/A':>16s}")
                rho_cells.append(f"{'N/A':>16s}")
                kappa_cells.append(f"{'N/A':>16s}")

        logging.info("    %-30s%s", "AbsDiff", "".join(abs_cells))
        logging.info("    %-30s%s", "Rho", "".join(rho_cells))
        logging.info("    %-30s%s", "Kappa", "".join(kappa_cells))
        if best_judge:
            logging.info(
                "    >> Best judge: %s (AbsDiff=%.3f)", best_judge, best_abs
            )

        # Per-judge per-score-column detail
        for judge in judge_names:
            merged = merge_on_material_id(
                h_cat, l_cat[l_cat["judge_id"] == judge], score_cols
            )
            if merged.empty:
                continue
            logging.info("\n      [%s / %s]", cat_value, judge)
            for col in score_cols:
                m = compute_agreement_metrics(
                    merged[f"{col}_h"], merged[f"{col}_l"]
                )
                if m:
                    logging.info(
                        "        %-28s"
                        " Rho=%6.3f Kappa=%6.3f AbsDiff=%6.3f n=%d",
                        col_label(col),
                        m["rho"],
                        m["kappa"],
                        m["abs_diff"],
                        m["n"],
                    )

    # Summary: best judge per category
    if grid:
        _section(
            f"BEST JUDGE PER {cat_label} (lowest AbsDiff on {OVERALL_COL})"
        )
        header = (
            f"{'Category':<35} {'Best Judge':<30}"
            f" {'AbsDiff':>7} {'Rho':>6} {'n':>4}"
        )
        logging.info("%s\n%s", header, "-" * len(header))
        for cat_value in sorted(grid):
            cat_metrics = grid[cat_value]
            best_j = min(
                cat_metrics, key=lambda j, cm=cat_metrics: cm[j]["abs_diff"]
            )
            best = grid[cat_value][best_j]
            logging.info(
                "%-35s %-30s %7.3f %6.3f %4d",
                cat_value,
                best_j,
                best["abs_diff"],
                best["rho"],
                best["n"],
            )

    if grid and len(grid) >= 2:
        _make_category_heatmap(grid, cat_values, judge_names, cat_label)

    return csv_rows


def _make_category_heatmap(grid, cat_values, judge_names, cat_label):
    """Generate a heatmap PNG showing AbsDiff per category x judge."""
    import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel
    import seaborn as sns  # pylint: disable=import-outside-toplevel

    cats_with_data = [c for c in cat_values if c in grid]
    matrix = np.full((len(cats_with_data), len(judge_names)), np.nan)
    for row_idx, cat in enumerate(cats_with_data):
        for col_idx, judge in enumerate(judge_names):
            if judge in grid.get(cat, {}):
                matrix[row_idx, col_idx] = grid[cat][judge]["abs_diff"]

    fig_height = max(6, len(cats_with_data) * 0.8 + 2)
    plt.figure(figsize=(10, fig_height))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn_r",
        xticklabels=judge_names,
        yticklabels=cats_with_data,
        cbar_kws={"label": "Mean Absolute Difference"},
        vmin=0,
        vmax=2,
        linewidths=0.5,
        linecolor="gray",
    )
    plt.xlabel("Judge LLM", fontsize=12, fontweight="bold")
    plt.ylabel(cat_label, fontsize=12, fontweight="bold")
    plt.title(
        f"{cat_label} x Judge LLM Agreement Heatmap\n"
        f"(Mean Absolute Difference - Overall Score)",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    safe_name = cat_label.lower().replace(" ", "_")
    out_png = os.path.join(OUTPUT_DIR, f"multi_llm_heatmap_{safe_name}.png")
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()
    logging.info("\nSaved %s heatmap to %s", cat_label, out_png)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(OUTPUT_DIR, "multi_llm_by_category.log"),
        level=logging.INFO,
        force=True,
        filemode="w",
    )

    parser = argparse.ArgumentParser(
        description="Compare multi-LLM results -- by category"
    )
    parser.add_argument("--annotations-dir", default="annotations/")
    args = parser.parse_args()

    # same as compare_human_judge_scores_by_category.py
    skip_folders = [
        "annotation_guide_catalysis",
        "f2f0828a5de4a3262edc73876809a9fe03ed6ff5",
        "2883daff26f16a13134a26ca5d366549a14fcc9c",
        "90233593a9aa72b4bacfdeadc20050ae6d4b88e1",
    ]
    human_df, llm_df = load_annotations(args.annotations_dir, skip_folders)
    if human_df.empty or llm_df.empty:
        logging.error("No data loaded.")
        return

    score_cols = [c for c in SCORE_COLUMNS if c in human_df.columns]
    logging.info("Loaded %d human / %d LLM rows", len(human_df), len(llm_df))

    analysis_by_score_category(human_df, llm_df, score_cols)
    analysis_by_group(
        human_df, llm_df, score_cols, "synth_llm", "SYNTHESIS LLM"
    )
    analysis_by_group(human_df, llm_df, score_cols, "judge_id", "JUDGE LLM")
    per_score_per_judge(human_df, llm_df, score_cols)

    # Stratified analysis by material categories
    csv_rows = []
    for cat_col, cat_label in [
        ("target_compound_type", "Target Compound Type"),
        ("synthesis_method", "Synthesis Method"),
    ]:
        rows = analyze_by_material_category(
            human_df, llm_df, score_cols, cat_col, cat_label
        )
        csv_rows.extend(rows)

    if csv_rows:
        csv_path = os.path.join(
            OUTPUT_DIR, "multi_llm_agreement_by_material_category.csv"
        )
        pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
        logging.info("\nSaved category analysis CSV to %s", csv_path)

    logging.info("\nFull log at %s/multi_llm_by_category.log", OUTPUT_DIR)


if __name__ == "__main__":
    main()
