"""Shared utility functions for evaluation scripts.

Common functions used across compare_human_judge_scores_*.py and
compare_multi_llm_results_*.py:

- Constants: SCORE_COLUMNS
- Score categorization: categorize_score
- ICC calculations: calculate_icc_absolute_agreement, calculate_icc_consistency
- Material name normalization: normalize_material_name
- Fuzzy matching: calculate_string_similarity, find_best_matches
- DataFrame helpers: merge_on_material_id, col_label, aggregate_human_scores_df
- Agreement metrics: compute_agreement_metrics, evaluate_agreement_by_criterion

Plot extraction metrics:
- parse_ground_truth_csv, match_series, series_coord_metrics, compare_extraction_to_gt
"""

import csv
import logging
import re
from difflib import SequenceMatcher
from typing import Tuple

import numpy as np
import pandas as pd
import pingouin as pg
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import cohen_kappa_score
from scipy.stats import permutation_test  



# =============================================================================
# Constants
# =============================================================================
SCORE_COLUMNS = [
    "structural_completeness_score",
    "material_extraction_score",
    "process_steps_score",
    "equipment_extraction_score",
    "conditions_extraction_score",
    "semantic_accuracy_score",
    "format_compliance_score",
    "overall_score",
]


# =============================================================================
# Score Categorization
# =============================================================================
def categorize_score(value):
    """Bin a continuous score (0-5 scale) into ordinal categories 0-4 for kappa.

    Bins: 0 (<=1), 1 (1,2], 2 (2,3], 3 (3,4], 4 (>4).
    """
    if value <= 1:
        return 0
    if value <= 2:
        return 1
    if value <= 3:
        return 2
    if value <= 4:
        return 3
    return 4


# =============================================================================
# ICC Calculations
# =============================================================================
def _icc_long_format(scores1, scores2):
    """Build long-format DataFrame for pingouin ICC."""
    n = len(scores1)
    df = pd.DataFrame({
        "subject": np.arange(n),
        "rater1": scores1,
        "rater2": scores2,
    })
    return pd.melt(df, id_vars="subject", var_name="rater", value_name="rating")


def calculate_icc_absolute_agreement(scores1, scores2):
    """ICC(2,1): two-way random, absolute agreement, single measure."""
    if len(scores1) < 5:
        return float("nan")
    try:
        long = _icc_long_format(scores1, scores2)
        tbl = pg.intraclass_corr(
            data=long, targets="subject", raters="rater", ratings="rating"
        )
        row = tbl[tbl["Type"] == "ICC2"]
        return float(row["ICC"].iloc[0]) if not row.empty else float("nan")
    except (ValueError, AssertionError, KeyError, IndexError):
        return float("nan")


def calculate_icc_consistency(scores1, scores2):
    """ICC(3,1): two-way mixed, consistency, single measure."""
    if len(scores1) < 5:
        return float("nan")
    try:
        long = _icc_long_format(scores1, scores2)
        tbl = pg.intraclass_corr(
            data=long, targets="subject", raters="rater", ratings="rating"
        )
        row = tbl[tbl["Type"] == "ICC3"]
        return float(row["ICC"].iloc[0]) if not row.empty else float("nan")
    except (ValueError, AssertionError, KeyError, IndexError):
        return float("nan")


# =============================================================================
# Material Name Normalization
# =============================================================================
def normalize_material_name(name):
    """Normalize material name for fuzzy matching.

    Steps: lowercase, standardize separators, remove common prefixes/suffixes,
    collapse whitespace.
    """
    if not name:
        return ""

    normalized = str(name).lower().strip()

    # Standardize separators (replace various dashes and slashes)
    normalized = re.sub(r"[-—−/−\\]", "-", normalized)  # noqa: RUF001

    # Remove common suffixes
    for suffix in [
        " single crystals", " crystals", " nanostructures", " nanoparticles",
        " nanorods", " nanowires", " nanoneedles", " nanocombs", " nanocrystals",
        " composite", " ceramics", " powders", " powder", " films", " thin films",
        " layers", " samples", " materials", " compounds", " structures",
    ]:
        if normalized.endswith(suffix):
            normalized = normalized[: -len(suffix)]

    # Remove common prefixes
    for prefix in [
        "synthesis of ", "preparation of ", "fabrication of ", "formation of ",
    ]:
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix):]

    # Collapse whitespace
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


# =============================================================================
# Fuzzy Matching
# =============================================================================
def calculate_string_similarity(str1, str2):
    """Similarity between two material name strings (0.0 to 1.0).

    Uses a weighted combination of SequenceMatcher ratio, word-overlap
    Jaccard, and substring matching after normalization.
    """
    if not str1 or not str2:
        return 0.0

    norm1 = normalize_material_name(str1)
    norm2 = normalize_material_name(str2)

    if not norm1 or not norm2:
        return 0.0
    if norm1 == norm2:
        return 1.0

    # SequenceMatcher
    seq_sim = SequenceMatcher(None, norm1, norm2).ratio()

    # Word-overlap Jaccard
    words1, words2 = set(norm1.split()), set(norm2.split())
    if words1 and words2:
        word_sim = len(words1 & words2) / len(words1 | words2)
    else:
        word_sim = 0.0

    # Substring matching
    sub_sim = 0.0
    if len(norm1) > 3 and len(norm2) > 3:
        if norm1 in norm2 or norm2 in norm1:
            sub_sim = 0.8
        else:
            max_len = max(len(norm1), len(norm2))
            for i in range(len(norm1) - 2):
                for j in range(i + 3, len(norm1) + 1):
                    substr = norm1[i:j]
                    if len(substr) >= 3 and substr in norm2:
                        sub_sim = max(sub_sim, len(substr) / max_len)

    return 0.4 * seq_sim + 0.4 * word_sim + 0.2 * sub_sim


def find_best_matches(human_materials, llm_materials, similarity_threshold=0.7):
    """Greedy best-match between two material name lists.

    Computes all pairwise similarities, sorts descending, then assigns
    greedily so each name is used at most once.

    Returns dict mapping human_material -> llm_material.
    """
    all_pairs = []
    for h_mat in human_materials:
        for l_mat in llm_materials:
            sim = calculate_string_similarity(h_mat, l_mat)
            if sim >= similarity_threshold:
                all_pairs.append((sim, h_mat, l_mat))

    all_pairs.sort(reverse=True)

    matches = {}
    used = set()
    for _sim, h_mat, l_mat in all_pairs:
        if h_mat not in matches and l_mat not in used:
            matches[h_mat] = l_mat
            used.add(l_mat)
    return matches


# =============================================================================
# DataFrame Helpers
# =============================================================================
def merge_on_material_id(human_df, llm_df, columns, suffixes=("_h", "_l")):
    """Inner-join human and LLM DataFrames on material_id."""
    return pd.merge(
        human_df[["material_id", *columns]],
        llm_df[["material_id", *columns]],
        on="material_id",
        suffixes=suffixes,
    )


def col_label(col_name):
    """Convert score column name to human-readable label."""
    return col_name.replace("_score", "").replace("_", " ").title()


def aggregate_human_scores_df(df):
    """Average scores from multiple human evaluators per material_id.

    Filters to evaluator_type == 'human', groups by material_id, returns
    the mean of all numeric (score) columns.
    """
    human_evals = df[df["evaluator_type"] == "human"]
    return human_evals.groupby("material_id").mean(numeric_only=True)


# =============================================================================
# Agreement Metrics
# =============================================================================
def compute_agreement_metrics(
    human_scores, llm_scores,
    use_permutation=False, n_resamples=10_000, random_state=42,
):
    """Compute agreement metrics between aligned human and LLM score arrays.

    Returns a dict with keys: rho, p, kappa, icc2, icc3, h_mean, h_median,
    h_std, l_mean, l_median, l_std, mean_diff, abs_diff, n.
    Returns None if fewer than 2 valid pairs.
    """
    paired = pd.DataFrame({"h": human_scores, "l": llm_scores}).dropna()
    if len(paired) < 2:
        return None

    h = paired["h"].to_numpy(float)
    l = paired["l"].to_numpy(float)
    finite = np.isfinite(h) & np.isfinite(l)
    h, l = h[finite], l[finite]

    # Spearman
    if h.size < 2 or np.unique(h).size < 2 or np.unique(l).size < 2:
        rho, p = np.nan, np.nan
    else:
        res = spearmanr(h, l, nan_policy="omit", alternative="two-sided")
        rho, p = float(res.statistic), float(res.pvalue)

        if use_permutation and h.size < 500:

            def _stat(x_perm):
                return spearmanr(
                    x_perm, l, nan_policy="omit", alternative="two-sided"
                ).statistic

            res_perm = permutation_test(
                (h,), _stat, permutation_type="pairings",
                n_resamples=n_resamples, alternative="two-sided",
                random_state=random_state,
            )
            p = float(res_perm.pvalue)
            logging.debug(
                "permutation p=%.6g, asymptotic p=%.6g, rho=%.4f, n=%d",
                p, float(res.pvalue), rho, h.size,
            )

    # Quadratic weighted Cohen's kappa
    try:
        kappa = cohen_kappa_score(
            paired["h"].apply(categorize_score),
            paired["l"].apply(categorize_score),
            weights="quadratic",
        )
    except ValueError:
        kappa = np.nan

    # ICC
    icc2 = calculate_icc_absolute_agreement(paired["h"], paired["l"])
    icc3 = calculate_icc_consistency(paired["h"], paired["l"])

    diff = paired["l"] - paired["h"]
    return {
        "rho": rho, "p": p, "kappa": kappa,
        "icc2": icc2, "icc3": icc3,
        "h_mean": paired["h"].mean(), "h_median": paired["h"].median(),
        "h_std": paired["h"].std(),
        "l_mean": paired["l"].mean(), "l_median": paired["l"].median(),
        "l_std": paired["l"].std(),
        "mean_diff": diff.mean(), "abs_diff": diff.abs().mean(),
        "n": len(paired),
    }


def evaluate_agreement_by_criterion(
    human_df, llm_df, score_columns,
    use_permutation=False, n_resamples=10_000, random_state=42,
):
    """Per-column agreement metrics between human and LLM DataFrames.

    Merges on material_id, then calls compute_agreement_metrics for each
    score column.

    Returns dict of {column_name: metrics_dict_or_None}.
    """
    score_columns = [
        c for c in score_columns
        if c in human_df.columns and c in llm_df.columns
    ]
    merged = merge_on_material_id(
        human_df, llm_df, score_columns, suffixes=("_human", "_llm"),
    )
    results = {}
    for col in score_columns:
        results[col] = compute_agreement_metrics(
            merged[f"{col}_human"], merged[f"{col}_llm"],
            use_permutation=use_permutation,
            n_resamples=n_resamples,
            random_state=random_state,
        )
    return results


# =============================================================================
# Plot Extraction Metrics
# =============================================================================

def parse_ground_truth_csv(csv_path):
    """Parse ground-truth CSV into {series_name: [(x, y), ...]}."""
    with open(csv_path, newline="") as f:
        rows = list(csv.reader(f))
    if len(rows) < 3:
        return {}
    header = rows[0]
    series_cols = []
    for i in range(0, len(header), 2):
        name = header[i].strip()
        if name:
            series_cols.append((name, i))
    result = {}
    for name, x_col in series_cols:
        y_col = x_col + 1
        coords = []
        for row in rows[2:]:
            if x_col < len(row) and y_col < len(row):
                xs, ys = row[x_col].strip(), row[y_col].strip()
                if xs and ys:
                    try:
                        coords.append((float(xs), float(ys)))
                    except ValueError:
                        pass
        if coords:
            result[name] = coords
    return result


def series_coord_metrics(ext_pts, gt_pts):
    """Nearest-neighbour RMSE/MAE and correlation between extracted and GT series.

    Distances normalized by GT axis ranges (0-1 scale).
    Pearson r, Spearman rho, and ICC are computed on the raw y-values
    paired via nearest-neighbour matching.
    """
    _empty = {"rmse_norm": None, "mae_norm": None,
              "pearson_r": None, "spearman_rho": None, "icc": None,
              "n_extracted": len(ext_pts), "n_gt": len(gt_pts)}
    if not ext_pts or not gt_pts:
        return _empty

    from scipy.spatial.distance import cdist

    ext = np.asarray(ext_pts)
    gt = np.asarray(gt_pts)

    # Normalize to 0-1 using GT axis ranges
    gt_min = gt.min(axis=0)
    gt_range = np.maximum(np.ptp(gt, axis=0), 1e-9)
    ext_n = (ext - gt_min) / gt_range
    gt_n = (gt - gt_min) / gt_range

    # Nearest-neighbour distances (GT → EXT)
    D = cdist(gt_n, ext_n)
    nn = D.min(axis=1)
    nn_idx = D.argmin(axis=1)

    # Correlation / ICC on paired raw y-values
    agreement = compute_agreement_metrics(gt[:, 1], ext[nn_idx, 1])
    if agreement is not None:
        sr = round(agreement["rho"], 4) if not np.isnan(agreement["rho"]) else None
        icc_val = round(agreement["icc2"], 4) if not np.isnan(agreement["icc2"]) else None
    else:
        sr, icc_val = None, None

    # Pearson r (not in compute_agreement_metrics)
    gt_y, ext_y = gt[:, 1], ext[nn_idx, 1]
    _enough = len(gt_y) >= 3 and np.unique(gt_y).size >= 2 and np.unique(ext_y).size >= 2
    pr = round(float(pearsonr(gt_y, ext_y).statistic), 4) if _enough else None

    return {
        "rmse_norm": round(float(np.sqrt(np.mean(nn ** 2))), 4),
        "mae_norm": round(float(np.mean(nn)), 4),
        "pearson_r": pr,
        "spearman_rho": sr,
        "icc": icc_val,
        "n_extracted": len(ext_pts),
        "n_gt": len(gt_pts),
    }


def compare_extraction_to_gt(extraction, gt_series, similarity_threshold=0.5):
    """Compare VLM extraction to ground truth.

    Uses ``find_best_matches`` and ``calculate_string_similarity`` for series
    name matching (reuses existing eval_utils fuzzy matching).

    Returns dict with status, num_series_*, mean_*_norm, matched_series[].
    """
    if not gt_series:
        return {"status": "no_gt"}
    if extraction is None:
        return {
            "status": "failed",
            "num_series_gt": len(gt_series),
            "num_series_extracted": 0,
            "num_series_matched": 0,
            "mean_rmse_norm": None,
            "mean_mae_norm": None,
            "mean_pearson_r": None,
            "mean_spearman_rho": None,
            "mean_icc": None,
            "matched_series": [],
        }
    ntc = extraction.get("name_to_coordinates", {})
    ext_names = list(ntc.keys())
    gt_names = list(gt_series.keys())

    # Reuse existing greedy fuzzy matcher (gt -> ext mapping)
    matches = find_best_matches(
        gt_names, ext_names, similarity_threshold=similarity_threshold,
    )
    # matches: {gt_name: ext_name}
    n_ext, n_gt, n_match = len(ext_names), len(gt_names), len(matches)

    series_detail, rmses, maes, prs, srs, iccs = [], [], [], [], [], []
    for gt_name, ext_name in matches.items():
        score = calculate_string_similarity(gt_name, ext_name)
        ext_coords = [(c[0], c[1]) for c in ntc[ext_name]]
        m = series_coord_metrics(ext_coords, gt_series[gt_name])
        series_detail.append({
            "extracted_name": ext_name,
            "gt_name": gt_name,
            "name_match_score": round(score * 100),
            **m,
        })
        if m["rmse_norm"] is not None:
            rmses.append(m["rmse_norm"])
            maes.append(m["mae_norm"])
        if m["pearson_r"] is not None:
            prs.append(m["pearson_r"])
        if m["spearman_rho"] is not None:
            srs.append(m["spearman_rho"])
        if m["icc"] is not None:
            iccs.append(m["icc"])

    return {
        "status": "ok",
        "num_series_gt": n_gt,
        "num_series_extracted": n_ext,
        "num_series_matched": n_match,
        "mean_rmse_norm": round(float(np.mean(rmses)), 4) if rmses else None,
        "mean_mae_norm": round(float(np.mean(maes)), 4) if maes else None,
        "mean_pearson_r": round(float(np.mean(prs)), 4) if prs else None,
        "mean_spearman_rho": round(float(np.mean(srs)), 4) if srs else None,
        "mean_icc": round(float(np.mean(iccs)), 4) if iccs else None,
        "matched_series": series_detail,
    }
