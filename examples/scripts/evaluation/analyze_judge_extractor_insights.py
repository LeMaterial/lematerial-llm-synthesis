"""Deeper insight metrics for the multi-LLM extraction/judge matrix.

Complements the agreement scripts (compare_multi_llm_results_*.py) by mining
the raw ``annotations/<id>/result.json`` (N extractors x M judges grid) and
``result_human.json`` (the human judge, one score per extractor) for effects
the agreement ranking does not surface:

1. Judge x extractor mean-score matrix (overall_score) + self-judging diagonal.
2. Self-preference bias ("grading your own homework"): how much a model, when
   judging its own extraction, deviates from what the *other* judges give the
   same extraction.
3. Judge leniency (mean score assigned) and scale usage (std) vs the human.
4. Extractor quality by the human and by the judges (with and without the
   extractor's own self-vote), plus coverage (materials produced).
5. Per-dimension means (which of the 8 rubric dimensions score lowest ->
   systematic extraction failure modes), globally and per extractor.
6. Inter-judge agreement (pairwise Spearman on overall_score).

All tables are written as CSVs under ``results/agreement_analysis/`` and echoed
to stdout. Cells with a null/missing ``evaluation`` payload are skipped, so no
folders need to be excluded manually.

Usage:
    uv run python \
        examples/scripts/evaluation/analyze_judge_extractor_insights.py
    uv run .../analyze_judge_extractor_insights.py --annotations-dir annotations
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_multi_llm_results_complete import (
    load_annotations,
)
from eval_utils import (
    compute_agreement_metrics,
    merge_on_material_id,
)

# Papers whose result.json carries null judge outputs (would crash the matched
# loader) or non-paper helper folders; matches the compare_* scripts.
SKIP_FOLDERS = [
    "annotation_guide_catalysis",
    "2883daff26f16a13134a26ca5d366549a14fcc9c",
    "90233593a9aa72b4bacfdeadc20050ae6d4b88e1",
]

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
OVERALL = "overall_score"
OUTPUT_DIR = "results/agreement_analysis"


def _scores_of(evaluation_wrapper):
    """Return the scores dict from an evaluations[] entry, or None if absent."""
    if not isinstance(evaluation_wrapper, dict):
        return None
    inner = evaluation_wrapper.get("evaluation")
    if not isinstance(inner, dict):
        return None
    scores = inner.get("scores")
    return scores if isinstance(scores, dict) else None


def load_long(annotations_dir):
    """Build tidy long DataFrames from raw result.json / result_human.json.

    Returns (llm_df, human_df):
      llm_df   rows: paper_id, material, synth_llm, judge_llm, <SCORE_COLUMNS>
      human_df rows: paper_id, material, synth_llm (=extractor), <SCORE_COLUMNS>
    """
    llm_rows, human_rows = [], []
    for paper_id in sorted(os.listdir(annotations_dir)):
        paper_dir = os.path.join(annotations_dir, paper_id)
        llm_path = os.path.join(paper_dir, "result.json")
        human_path = os.path.join(paper_dir, "result_human.json")
        if not os.path.isdir(paper_dir):
            continue

        # --- LLM judges (N x M grid) ---
        if os.path.exists(llm_path):
            with open(llm_path, encoding="utf-8") as fh:
                llm_data = json.load(fh)
            for entry in llm_data:
                synth_llm = entry.get("synth_llm", "")
                for mat in entry.get("materials", []):
                    name = mat.get("material", "")
                    synth = mat.get("synthesis", {}) or {}
                    if "Extraction failed:" in str(synth.get("notes") or ""):
                        continue
                    for ev in mat.get("evaluations", []):
                        scores = _scores_of(ev)
                        if scores is None:
                            continue
                        row = {
                            "paper_id": paper_id,
                            "material": name,
                            "synth_llm": synth_llm,
                            "judge_llm": ev.get("judge_llm", ""),
                        }
                        for c in SCORE_COLUMNS:
                            v = scores.get(c)
                            num = isinstance(v, (int, float))
                            row[c] = v if num else np.nan
                        llm_rows.append(row)

        # --- Human judge (one score per extractor, positional) ---
        if os.path.exists(human_path):
            with open(human_path, encoding="utf-8") as fh:
                human_data = json.load(fh)
            order = human_data.get("extractor_order", []) or []
            for mat in human_data.get("materials", []):
                name = mat.get("material_name", "")
                evals = mat.get("evaluations", []) or []
                for idx, synth_llm in enumerate(order):
                    if idx >= len(evals):
                        continue
                    scores = _scores_of(evals[idx])
                    if scores is None:
                        continue
                    row = {
                        "paper_id": paper_id,
                        "material": name,
                        "synth_llm": synth_llm,
                    }
                    for c in SCORE_COLUMNS:
                        v = scores.get(c)
                        row[c] = v if isinstance(v, (int, float)) else np.nan
                    human_rows.append(row)

    return pd.DataFrame(llm_rows), pd.DataFrame(human_rows)


def judge_extractor_matrix(llm_df):
    """Mean overall_score per (synth_llm row, judge_llm col)."""
    return llm_df.pivot_table(
        index="synth_llm", columns="judge_llm", values=OVERALL, aggfunc="mean"
    ).round(3)


def self_preference(llm_df):
    """Self-favoritism: (model judging its own extraction) minus (mean of the
    other judges on the same extraction), averaged over the model's extractions.

    Positive => the model inflates its own outputs relative to its peers.
    """
    rows = []
    for model in sorted(set(llm_df["synth_llm"]) | set(llm_df["judge_llm"])):
        own = llm_df[
            (llm_df["synth_llm"] == model) & (llm_df["judge_llm"] == model)
        ]
        diffs, self_scores, peer_scores = [], [], []
        for _, r in own.iterrows():
            peers = llm_df[
                (llm_df["paper_id"] == r["paper_id"])
                & (llm_df["material"] == r["material"])
                & (llm_df["synth_llm"] == model)
                & (llm_df["judge_llm"] != model)
            ]
            if peers.empty or not np.isfinite(r[OVERALL]):
                continue
            peer_mean = peers[OVERALL].mean()
            diffs.append(r[OVERALL] - peer_mean)
            self_scores.append(r[OVERALL])
            peer_scores.append(peer_mean)
        rows.append(
            {
                "model": model,
                "n_self_cells": len(diffs),
                "self_score_mean": np.mean(self_scores) if diffs else np.nan,
                "peer_score_mean": np.mean(peer_scores) if diffs else np.nan,
                "self_preference": np.mean(diffs) if diffs else np.nan,
            }
        )
    return pd.DataFrame(rows).round(3)


def self_bias_did(matrix):
    """Difference-in-differences self-bias from the judge x extractor matrix.

    For model M (using overall-score cell means):
      self_lift  = M(own extraction) - mean(M judging other extractions)
      peer_lift  = mean(peers judging M's extraction)
                   - mean(peers judging other extractions)
      did        = self_lift - peer_lift

    ``did`` removes both M's leniency (via self_lift, within M's own column)
    and M's extraction quality (via peer_lift), isolating whether M treats its
    *own* homework more favourably than the field does.
    """
    models = list(matrix.index)
    rows = []
    for m in models:
        others = [x for x in models if x != m]
        self_lift = matrix.loc[m, m] - matrix.loc[others, m].mean()
        peer_lift = (
            matrix.loc[m, others].mean()
            - matrix.loc[others, others].to_numpy().mean()
        )
        rows.append(
            {
                "model": m,
                "self_lift": round(self_lift, 3),
                "peer_lift": round(peer_lift, 3),
                "self_bias_did": round(self_lift - peer_lift, 3),
            }
        )
    return pd.DataFrame(rows)


def judge_behavior(llm_df, human_df):
    """Per-judge leniency (mean), spread (std) and per-dimension means."""
    rows = []
    for judge in sorted(llm_df["judge_llm"].dropna().unique()):
        sub = llm_df[llm_df["judge_llm"] == judge]
        row = {
            "judge": judge,
            "overall_mean": sub[OVERALL].mean(),
            "overall_std": sub[OVERALL].std(),
            "n": sub[OVERALL].notna().sum(),
        }
        for c in SCORE_COLUMNS:
            row[c] = sub[c].mean()
        rows.append(row)
    # human reference row
    hrow = {
        "judge": "HUMAN",
        "overall_mean": human_df[OVERALL].mean(),
        "overall_std": human_df[OVERALL].std(),
        "n": human_df[OVERALL].notna().sum(),
    }
    for c in SCORE_COLUMNS:
        hrow[c] = human_df[c].mean()
    rows.append(hrow)
    return pd.DataFrame(rows).round(3)


def extractor_quality(llm_df, human_df):
    """Per-extractor mean overall by human, by all judges, and by peer judges
    (excluding the extractor's own self-vote), plus coverage."""
    rows = []
    extractors = sorted(
        set(llm_df["synth_llm"].dropna()) | set(human_df["synth_llm"].dropna())
    )
    for ex in extractors:
        j = llm_df[llm_df["synth_llm"] == ex]
        j_peer = j[j["judge_llm"] != ex]
        h = human_df[human_df["synth_llm"] == ex]
        rows.append(
            {
                "extractor": ex,
                "human_overall": h[OVERALL].mean(),
                "judge_overall_all": j[OVERALL].mean(),
                "judge_overall_peer": j_peer[OVERALL].mean(),
                "n_materials_human": h[["paper_id", "material"]]
                .drop_duplicates()
                .shape[0],
                "n_materials_llm": j[["paper_id", "material"]]
                .drop_duplicates()
                .shape[0],
            }
        )
    return pd.DataFrame(rows).round(3)


def dimension_means(llm_df, human_df):
    """Global mean per rubric dimension (judges pooled) and human, to expose
    which dimensions systematically score lowest (failure modes)."""
    rows = []
    for c in SCORE_COLUMNS:
        rows.append(
            {
                "dimension": c.replace("_score", ""),
                "judges_mean": llm_df[c].mean(),
                "human_mean": human_df[c].mean(),
            }
        )
    return pd.DataFrame(rows).round(3)


def interjudge_agreement(llm_df):
    """Pairwise Spearman between judges on overall_score over shared
    (paper, material, synth_llm) cells."""
    judges = sorted(llm_df["judge_llm"].dropna().unique())
    wide = llm_df.pivot_table(
        index=["paper_id", "material", "synth_llm"],
        columns="judge_llm",
        values=OVERALL,
        aggfunc="mean",
    )
    mat = pd.DataFrame(index=judges, columns=judges, dtype=float)
    for a in judges:
        for b in judges:
            if a == b:
                mat.loc[a, b] = 1.0
                continue
            pair = wide[[a, b]].dropna()
            ca, cb = pair.iloc[:, 0], pair.iloc[:, 1]
            if len(pair) >= 3 and ca.nunique() > 1 and cb.nunique() > 1:
                mat.loc[a, b] = spearmanr(ca, cb).statistic
    return mat.round(3)


def judge_ranking_variants(human_df, llm_df):
    """Human-vs-judge agreement on overall_score for three cell sets:

      all         - every matched cell (the headline ranking)
      loo_no_self - cells where the judge did NOT extract the material
                    (leave-one-out: removes any self-evaluation conflict)
      self_only   - cells where the judge graded its own extraction

    If a judge's rank is unchanged between ``all`` and ``loo_no_self``, its
    standing does not depend on grading its own work.
    """
    rows = []
    for judge in sorted(llm_df["judge_id"].dropna().unique()):
        jdf = llm_df[llm_df["judge_id"] == judge]
        variants = {
            "all": jdf,
            "loo_no_self": jdf[jdf["synth_llm"] != judge],
            "self_only": jdf[jdf["synth_llm"] == judge],
        }
        for name, sub in variants.items():
            merged = merge_on_material_id(human_df, sub, ["overall_score"])
            m = compute_agreement_metrics(
                merged["overall_score_h"], merged["overall_score_l"]
            )
            if not m:
                continue
            rows.append(
                {
                    "judge": judge,
                    "cell_set": name,
                    "n": m["n"],
                    "rho": round(m["rho"], 3),
                    "kappa": round(m["kappa"], 3),
                    "icc2": round(m["icc2"], 3),
                    "icc3": round(m["icc3"], 3),
                    "abs_diff": round(m["abs_diff"], 3),
                    "mean_diff": round(m["mean_diff"], 3),
                }
            )
    df = pd.DataFrame(rows)
    # rank judges within each cell_set by icc2 (desc) for easy comparison
    df["rank_icc2"] = df.groupby("cell_set")["icc2"].rank(
        ascending=False, method="min"
    ).astype(int)
    return df.sort_values(["cell_set", "rank_icc2"]).reset_index(drop=True)


def extractor_ranking_by_judge(matrix, human_df):
    """Does each judge reproduce the human's ranking of the extractors?

    Compares each judge's per-extractor mean overall_score (a column of the
    judge x extractor matrix) against the human's per-extractor mean, via
    Spearman over the extractors, and records each judge's top-ranked extractor.
    """
    human_vec = human_df.groupby("synth_llm")["overall_score"].mean()
    extractors = list(matrix.index)
    human_vec = human_vec.reindex(extractors)
    human_order = list(human_vec.sort_values(ascending=False).index)
    rows = [
        {
            "grader": "HUMAN",
            "top_extractor": human_order[0],
            "ranking": " > ".join(human_order),
            "spearman_vs_human": 1.0,
        }
    ]
    for judge in matrix.columns:
        col = matrix[judge].reindex(extractors)
        order = list(col.sort_values(ascending=False).index)
        rho = spearmanr(col.to_numpy(), human_vec.to_numpy()).statistic
        rows.append(
            {
                "grader": judge,
                "top_extractor": order[0],
                "ranking": " > ".join(order),
                "spearman_vs_human": round(float(rho), 3),
            }
        )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations-dir", default="annotations")
    args = parser.parse_args()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    llm_df, human_df = load_long(args.annotations_dir)

    def dump(df, name, index=False):
        path = os.path.join(OUTPUT_DIR, name)
        df.to_csv(path, index=index)
        print(f"\n===== {name} =====")
        print(df.to_string(index=index))
        return path

    print(
        f"Loaded {len(llm_df)} LLM-judge score rows and "
        f"{len(human_df)} human score rows."
    )
    matrix = judge_extractor_matrix(llm_df)
    dump(matrix, "insights_judge_extractor_matrix.csv", index=True)
    dump(self_preference(llm_df), "insights_self_preference.csv")
    dump(self_bias_did(matrix), "insights_self_bias_did.csv")
    dump(judge_behavior(llm_df, human_df), "insights_judge_behavior.csv")
    dump(extractor_quality(llm_df, human_df), "insights_extractor_quality.csv")
    dump(dimension_means(llm_df, human_df), "insights_dimension_means.csv")
    dump(interjudge_agreement(llm_df), "insights_interjudge_spearman.csv",
         index=True)
    dump(extractor_ranking_by_judge(matrix, human_df),
         "insights_extractor_ranking_by_judge.csv")

    # matched human<->judge pairs (same pipeline as the agreement ranking)
    m_human_df, m_llm_df = load_annotations(args.annotations_dir, SKIP_FOLDERS)
    dump(judge_ranking_variants(m_human_df, m_llm_df),
         "insights_judge_ranking_loo.csv")
    print(f"\nAll insight CSVs written to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
