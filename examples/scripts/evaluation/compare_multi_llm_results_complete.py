"""Compare multi-LLM result.json with result_human.json -- rank judges based on the agreement with human score"""

import argparse
import json
import logging
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_utils import (  # pylint: disable=wrong-import-position
    SCORE_COLUMNS,
    col_label,
    compute_agreement_metrics,
    merge_on_material_id,
    normalize_material_name,
    find_best_matches,
)

OUTPUT_DIR = "results/agreement_analysis"


def load_annotations(annotations_dir, skip_folders=None):
    """Load and align human vs LLM-judge evaluations.

    Returns:
        (human_df, llm_df) DataFrames with columns including SCORE_COLUMNS,
        material_id, paper_id, synth_llm, judge_id, target_compound_type,
        and synthesis_method.
    """
    skip_folders = skip_folders or []
    human_rows, llm_rows = [], []
    processed_papers, skipped_papers, skipped_extractions = [], [], []

    for paper_id in sorted(os.listdir(annotations_dir)):
        paper_dir = os.path.join(annotations_dir, paper_id)
        if not os.path.isdir(paper_dir) or paper_id in skip_folders:
            skipped_papers.append(paper_id)
            continue

        llm_path = os.path.join(paper_dir, "result.json")
        human_path = os.path.join(paper_dir, "result_human.json")
        if not (os.path.exists(llm_path) and os.path.exists(human_path)):
            skipped_papers.append(paper_id)
            continue

        try:
            with open(llm_path, encoding="utf-8") as fh:
                llm_data = json.load(fh)
            with open(human_path, encoding="utf-8") as fh:
                human_data = json.load(fh)
        except (json.JSONDecodeError, KeyError) as exc:
            logging.info("Error reading files for %s: %s", paper_id, exc)
            skipped_papers.append(f"{paper_id} (file read error)")
            continue

        processed_papers.append(paper_id)
        n_human_mats = len(human_data.get("materials", []))
        n_synth_llms = len(llm_data)
        logging.info(
            "Processing %s: %d human materials, %d synth LLMs",
            paper_id, n_human_mats, n_synth_llms,
        )

        extractor_order = human_data.get("extractor_order", [])

        # --- Index LLM materials, judge scores, and raw name mapping ---
        judge_scores_lookup = {}   # (synth_llm, normalized_name) -> {judge: scores}
        raw_name_map = {}          # (synth_llm, normalized_name) -> raw material name - to map back 
                                   # original material name during logging
        for entry in llm_data:
            synth_llm = entry.get("synth_llm", "")
            for mat_entry in entry.get("materials", []):
                mat_name = mat_entry.get("material", "")
                synth_info = mat_entry.get("synthesis", {})
                if "Extraction failed:" in str(synth_info.get("notes", "") or ""):
                    skipped_extractions.append(
                        f"{paper_id}/{synth_llm}/{mat_name} (extraction failed)"
                    )
                    continue
                norm_key = (synth_llm, normalize_material_name(mat_name))
                raw_name_map[norm_key] = mat_name
                judge_scores_lookup[norm_key] = {
                    evaluation.get("judge_llm", ""): evaluation.get("evaluation", {}).get("scores", {})
                    for evaluation in mat_entry.get("evaluations", [])
                }

        # --- Collect human scores per synth LLM (first pass) ---
        human_scores_by_synth = {}  # synth_llm -> {material_name: scores_dict}
        for human_mat in human_data.get("materials", []):
            mat_name = human_mat.get("material_name", "")
            evals = human_mat.get("evaluations", [])
            for idx, synth_llm in enumerate(extractor_order):
                if idx >= len(evals):
                    continue
                scores = evals[idx].get("evaluation", {}).get("scores", {})
                if not any(scores.get(c) is not None for c in SCORE_COLUMNS):
                    continue
                human_scores_by_synth.setdefault(synth_llm, {})[mat_name] = scores

        # --- Greedy best-match per synth LLM (threshold 0.7) ---
        match_map = {}  # synth_llm -> {human_name: (synth_llm, norm_llm_name)}
        for synth_llm, h_scores in human_scores_by_synth.items():
            h_names = list(h_scores.keys())
            norm_h = [normalize_material_name(n) for n in h_names]
            norm_l = [nk for (sl, nk) in judge_scores_lookup if sl == synth_llm]
            matches = find_best_matches(norm_h, norm_l, similarity_threshold=0.7)
            norm_to_orig = {normalize_material_name(h_name): h_name for h_name in h_names}
            match_map[synth_llm] = {
                norm_to_orig[norm_human_name]: (synth_llm, norm_llm_name)
                for norm_human_name, norm_llm_name in matches.items() if norm_human_name in norm_to_orig
            }

        # --- Build DataFrame rows (second pass) ---
        matched_by_synth = {}
        human_only_by_synth = {}
        matched_llm_keys = set()

        for synth_llm, h_scores in human_scores_by_synth.items():
            matched_by_synth[synth_llm] = []
            human_only_by_synth[synth_llm] = []

            for mat_name, scores in h_scores.items():
                material_id = f"{paper_id}__{synth_llm}__{mat_name}"
                base = {
                    "paper_id": paper_id,
                    "material_id": material_id,
                    "material": mat_name,
                    "synth_llm": synth_llm,
                }
                lookup_key = match_map.get(synth_llm, {}).get(mat_name)

                human_rows.append({
                    **base, "judge_id": "human",
                    **{c: scores.get(c) for c in SCORE_COLUMNS},
                })

                if not lookup_key:
                    human_only_by_synth[synth_llm].append(mat_name)
                    continue

                matched_norm = lookup_key[1]
                orig_llm_name = raw_name_map.get(lookup_key, matched_norm)
                display = (
                    mat_name
                    if normalize_material_name(mat_name) == matched_norm
                    else f"{mat_name} -> {orig_llm_name}"
                )
                matched_by_synth[synth_llm].append(display)
                matched_llm_keys.add(lookup_key)

                for judge_llm, j_scores in judge_scores_lookup[lookup_key].items():
                    llm_rows.append({
                        **base, "judge_id": judge_llm,
                        **{c: j_scores.get(c) for c in SCORE_COLUMNS},
                    })

        # --- Identify unmatched LLM-only materials ---
        llm_only_by_synth = {}
        for (sl, nk), raw in raw_name_map.items():
            if (sl, nk) not in matched_llm_keys:
                llm_only_by_synth.setdefault(sl, []).append(raw)

        # --- Log matching details per synth LLM ---
        all_synths = set(matched_by_synth) | set(human_only_by_synth) | set(llm_only_by_synth)
        for synth_llm in sorted(all_synths):
            matched = matched_by_synth.get(synth_llm, [])
            human_only = human_only_by_synth.get(synth_llm, [])
            llm_only = llm_only_by_synth.get(synth_llm, [])
            total_human = len(matched) + len(human_only)
            if not (matched or human_only or llm_only):
                continue
            if total_human == 0:
                logging.info("  [%s] No human evaluations", synth_llm)
            else:
                logging.info(
                    "  [%s] %d/%d human materials matched",
                    synth_llm, len(matched), total_human,
                )
            if matched:
                logging.info("    Matched: %s", matched)
            if human_only:
                logging.info("    Unmatched (human-only): %s", human_only)
            if llm_only:
                logging.info("    Unmatched (llm-only): %s", llm_only)

    # --- Summary ---
    logging.info(
        "\nProcessed %d papers with both human and LLM evaluations:",
        len(processed_papers),
    )
    for paper in processed_papers:
        logging.info("  - %s", paper)
    if skipped_papers:
        logging.info("\nSkipped %d papers:", len(skipped_papers))
        for paper in skipped_papers:
            logging.info("  - %s", paper)
    if skipped_extractions:
        logging.info(
            "\nSkipped %d materials (extraction failures):", len(skipped_extractions)
        )
        for item in skipped_extractions:
            logging.info("  - %s", item)
    logging.info("\nTotal human rows: %d | Total LLM rows: %d", len(human_rows), len(llm_rows))

    return pd.DataFrame(human_rows), pd.DataFrame(llm_rows)


def log_individual_scores(human_df, llm_df, score_cols):
    """Log side-by-side human vs judge scores for every matched material."""
    merged = pd.merge(
        human_df[["material_id", "paper_id", "material", "synth_llm", *score_cols]],
        llm_df[["material_id", "judge_id", *score_cols]],
        on="material_id",
        suffixes=("_h", "_l"),
    )
    logging.info("\n%s\nINDIVIDUAL SCORE COMPARISONS\n%s", "=" * 100, "=" * 100)
    for _, row in merged.iterrows():
        logging.info(
            "\nMaterial: %s | Synth: %s | Judge: %s | Paper: %s",
            row["material"], row["synth_llm"], row["judge_id"], row["paper_id"],
        )
        for col in score_cols:
            h_val, l_val = row.get(f"{col}_h"), row.get(f"{col}_l")
            if h_val is not None and l_val is not None:
                logging.info(
                    "  %-28s  Human: %5.1f | LLM: %5.1f | Diff: %+5.1f",
                    col_label(col), h_val, l_val, l_val - h_val,
                )


def rank_judges(human_df, llm_df, _score_cols=None, rank_by="abs_diff"):
    """Rank judges by alignment with human on overall_score.

    Returns a list of dicts sorted by *rank_by* (ascending for abs_diff,
    descending for correlation-like metrics).
    """
    overall = "overall_score"
    results = []
    for judge in sorted(llm_df["judge_id"].dropna().unique()):
        merged = merge_on_material_id(human_df, llm_df[llm_df["judge_id"] == judge], [overall])
        metrics = compute_agreement_metrics(merged[f"{overall}_h"], merged[f"{overall}_l"])
        if metrics:
            results.append({"judge": judge, **metrics})
    descending = rank_by != "abs_diff"
    return sorted(
        results,
        key=lambda r: (-r[rank_by] if descending else r[rank_by], r["abs_diff"]),
    )


def _log_table(metrics_by_col, score_cols):
    """Log a formatted table of agreement metrics per score column."""
    header = (
        f"{'Criterion':<28} {'Rho':>6} {'p':>8} {'Kappa':>7} "
        f"{'ICC2':>7} {'ICC3':>7} {'H-Mean':>7} {'H-Med':>6} {'H-Std':>6} "
        f"{'L-Mean':>7} {'L-Med':>6} {'L-Std':>6} {'n':>4}"
    )
    logging.info("%s\n%s", header, "-" * len(header))
    for col in score_cols:
        m = metrics_by_col.get(col)
        if m:
            logging.info(
                "%-28s %6.3f %8.4f %7.3f %7.3f %7.3f "
                "%7.2f %6.2f %6.2f %7.2f %6.2f %6.2f %4d",
                col_label(col), m["rho"], m["p"], m["kappa"],
                m["icc2"], m["icc3"],
                m["h_mean"], m["h_median"], m["h_std"],
                m["l_mean"], m["l_median"], m["l_std"], m["n"],
            )


def log_judge_ranking(ranked, score_cols, human_df, llm_df, rank_by="abs_diff"):
    """Log the ranked judge table and per-judge score breakdowns."""
    logging.info("\n%s", "=" * 100)
    logging.info(
        "JUDGE LLM RANKING -- closest to human (overall_score, ranked by %s)", rank_by
    )
    logging.info("%s", "=" * 100)
    header = (
        f"{'Rank':<5} {'Judge LLM':<30} {'Rho':>6} {'p':>8} {'Kappa':>7} "
        f"{'ICC2':>7} {'ICC3':>7} {'H-Mean':>7} {'H-Std':>6} "
        f"{'J-Mean':>7} {'J-Std':>6} {'MeanDiff':>8} {'AbsDiff':>7} {'n':>4}"
    )
    logging.info("%s\n%s", header, "-" * len(header))
    prefix = "* " if rank_by == "abs_diff" else ""
    for rank, entry in enumerate(ranked, 1):
        logging.info(
            "%s%-5d %-30s %6.3f %8.4f %7.3f %7.3f %7.3f "
            "%7.2f %6.2f %7.2f %6.2f %8.3f %7.3f %4d",
            prefix, rank, entry["judge"],
            entry["rho"], entry["p"], entry["kappa"],
            entry["icc2"], entry["icc3"],
            entry["h_mean"], entry["h_std"],
            entry["l_mean"], entry["l_std"],
            entry["mean_diff"], entry["abs_diff"], entry["n"],
        )

    logging.info("\n%s\nPER-JUDGE x PER-CATEGORY BREAKDOWN\n%s", "=" * 100, "=" * 100)
    for rank, entry in enumerate(ranked, 1):
        logging.info(
            "\n--- Judge: %s (rank %d, AbsDiff=%.3f) ---",
            entry["judge"], rank, entry["abs_diff"],
        )
        merged = merge_on_material_id(human_df, llm_df[llm_df["judge_id"] == entry["judge"]], score_cols)
        _log_table(
            {c: compute_agreement_metrics(merged[f"{c}_h"], merged[f"{c}_l"]) for c in score_cols},
            score_cols,
        )


def synth_judge_heatmap(human_df, llm_df):
    """Generate and save a synth-LLM x judge-LLM heatmap on overall_score."""
    import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel
    import seaborn as sns  # pylint: disable=import-outside-toplevel

    overall = "overall_score"
    synth_llms = sorted(human_df["synth_llm"].dropna().unique())
    judge_llms = sorted(llm_df["judge_id"].dropna().unique())

    logging.info("\n%s", "=" * 100)
    logging.info("SYNTH LLM x JUDGE LLM - Mean AbsDiff (%s)", overall)
    logging.info("%s", "=" * 100)
    header = f"{'':>25s}" + "".join(f" {j:>16s}" for j in judge_llms)
    logging.info("%s\n%s", header, "-" * len(header))

    grid = {}
    for sl in synth_llms:
        cells = []
        for jl in judge_llms:
            merged = merge_on_material_id(
                human_df[human_df["synth_llm"] == sl],
                llm_df[(llm_df["synth_llm"] == sl) & (llm_df["judge_id"] == jl)],
                [overall],
            )
            metrics = compute_agreement_metrics(
                merged[f"{overall}_h"], merged[f"{overall}_l"]
            )
            if metrics:
                cells.append(f"{metrics['abs_diff']:>16.3f}")
                grid[(sl, jl)] = metrics["abs_diff"]
            else:
                cells.append(f"{'N/A':>16s}")
        logging.info("%25s%s", sl, "".join(cells))

    matrix = np.full((len(synth_llms), len(judge_llms)), np.nan)
    for row_idx, sl in enumerate(synth_llms):
        for col_idx, jl in enumerate(judge_llms):
            if (sl, jl) in grid:
                matrix[row_idx, col_idx] = grid[(sl, jl)]

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        matrix, annot=True, fmt=".3f", cmap="RdYlGn_r",
        xticklabels=judge_llms, yticklabels=synth_llms,
        cbar_kws={"label": "Mean Absolute Difference"},
        vmin=0, vmax=2, linewidths=0.5, linecolor="gray",
    )
    plt.xlabel("Judge LLM", fontsize=12, fontweight="bold")
    plt.ylabel("Synthesis LLM", fontsize=12, fontweight="bold")
    plt.title(
        "Synthesis LLM x Judge LLM Agreement Heatmap\n"
        "(Mean Absolute Difference - Overall Score)",
        fontsize=14, fontweight="bold", pad=20,
    )
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    out_png = os.path.join(OUTPUT_DIR, "multi_llm_heatmap_synth_judge.png")
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()
    logging.info("\nSaved heatmap to %s", out_png)


def _save_ranking_png(ranked, rank_by):
    """Render a small PNG table showing judge ranking."""
    import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel

    metric_map = {
        "abs_diff": ("AbsDiff", "abs_diff"),
        "rho": ("Rho", "rho"),
        "kappa": ("Kappa", "kappa"),
        "icc2": ("ICC(2,1)", "icc2"),
        "icc3": ("ICC(3,1)", "icc3"),
    }
    metric_label, metric_key = metric_map.get(rank_by, ("AbsDiff", "abs_diff"))
    headers = ["Rank", "Judge LLM", metric_label]
    table_data = [
        [str(i), entry["judge"], f"{entry[metric_key]:.3f}"]
        for i, entry in enumerate(ranked, 1)
    ]

    _fig, ax = plt.subplots(figsize=(12, len(ranked) * 1.2 + 2))
    ax.axis("off")
    tbl = ax.table(
        cellText=table_data, colLabels=headers,
        cellLoc="center", loc="center", bbox=[0, 0, 1, 1],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
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
        f"Judge LLM Ranking by Human Alignment (Overall Score)\nRanked by: {rank_by}",
        fontsize=14, fontweight="bold", pad=20,
    )
    out_png = os.path.join(OUTPUT_DIR, "multi_llm_judge_ranking.png")
    plt.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    return out_png




def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(OUTPUT_DIR, "multi_llm_complete.log"),
        level=logging.INFO, force=True, filemode="w",
    )
    parser = argparse.ArgumentParser(
        description="Compare multi-LLM results with human annotations",
    )
    parser.add_argument("--annotations-dir", default="annotations/")
    parser.add_argument(
        "--rank-by", default="abs_diff",
        choices=["abs_diff", "rho", "kappa", "icc2", "icc3"],
        help="Metric to rank judges by (default: abs_diff)",
    )
    args = parser.parse_args()

    #same as compare_human_judge_scores_complete.py
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

    # Aggregate multiple human evaluators to consensus scores
    human_counts = human_df.groupby("material_id").size()
    if (human_counts > 1).any():
        logging.info("\nMultiple human evaluators detected. Aggregating to consensus...")
        human_df = human_df.groupby("material_id", as_index=False)[score_cols].mean()

    log_individual_scores(human_df, llm_df, score_cols)

    ranked = rank_judges(human_df, llm_df, score_cols, rank_by=args.rank_by)
    if not ranked:
        logging.error("No judge results computed.")
        return

    log_judge_ranking(ranked, score_cols, human_df, llm_df, rank_by=args.rank_by)
    synth_judge_heatmap(human_df, llm_df)

    # Save JSON
    out_json = os.path.join(OUTPUT_DIR, "multi_llm_judge_ranking.json")
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(
            [{"rank": i + 1, "rank_by": args.rank_by, **entry}
             for i, entry in enumerate(ranked)],
            fh, indent=2, default=str,
        )

    # Save PNG ranking table
    out_png = _save_ranking_png(ranked, args.rank_by)

    logging.info(
        "\nSaved judge ranking to %s and %s | Log at %s/multi_llm_complete.log",
        out_json, out_png, OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()
