"""Regenerate every verifiable LaTeX table in supp.tex from source data.

Each table in supp.tex should trace to a script, not a manual transcription.
This script is that trace: it reads only from committed source files
(annotations/, results/agreement_analysis/, data/results_catalysis_*) and the
published LeMaterial/LeMat-Synth "full" config, and prints ready-to-paste
\\begin{tabular}...\\end{tabular} blocks, or writes them to
results/agreement_analysis/tables/*.tex if --write is passed.

Covers (see README table below for the supp.tex label each maps to):
  1. table:human-llm-comparison               -- judge agreement, loo_no_self
  2. worked examples (Bi-Pb-Sr-Cu-O, WFe2Ni-red, ...) -- per-judge verdicts
  3. tab:thermocat-vlm-extraction              -- VLM figure-extraction
     benchmark
  4. table:llm-syn-scores-synthesis-type / -material-type -- per-method/
     per-category mean +/- std judge scores, computed over the full
     LeMat-Synth "full" config (all procedures with a non-null evaluation).

Usage:
    uv run python examples/scripts/evaluation/generate_supp_tables.py
    uv run python examples/scripts/evaluation/generate_supp_tables.py --write
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(
    0, str(REPO_ROOT / "examples" / "scripts" / "case_study_thermocatalysis")
)

AGREEMENT_DIR = REPO_ROOT / "results" / "agreement_analysis"
ANNOTATIONS_DIR = REPO_ROOT / "annotations"
CATALYSIS_GT_DIR = REPO_ROOT / "data" / "results_catalysis_human"
CATALYSIS_RESULTS_DIR = REPO_ROOT / "data" / "results_catalysis_string_match"
CATALYSIS_LLM_MATCH_DIR = REPO_ROOT / "data" / "results_catalysis_llm_match"

TABLES_OUT_DIR = AGREEMENT_DIR / "tables"

MODEL_DISPLAY = {
    "claude-sonnet-4.6": "Claude-Sonnet-4.6",
    "deepseek-v3.2": "DeepSeek-V3.2",
    "gemini-3-flash": "Gemini-3-Flash",
    "qwen3.5-397b-a17b": "Qwen3.5-397B-A17B",
}


def _write_or_print(name: str, body: str, write: bool) -> None:
    if write:
        TABLES_OUT_DIR.mkdir(parents=True, exist_ok=True)
        path = TABLES_OUT_DIR / f"{name}.tex"
        path.write_text(body)
        print(f"wrote {path}")
    else:
        print(f"\n% ===== {name} =====")
        print(body)


# ---------------------------------------------------------------------------
# 1. table:human-llm-comparison (supp.tex ~line 378)
# ---------------------------------------------------------------------------


def generate_judge_agreement_table(write: bool = False) -> None:
    loo = pd.read_csv(AGREEMENT_DIR / "insights_judge_ranking_loo.csv")
    loo_no_self = loo[loo["cell_set"] == "loo_no_self"].sort_values(
        "icc2", ascending=False
    )

    rows = []
    for _, r in loo_no_self.iterrows():
        rows.append(
            f"    {MODEL_DISPLAY[r['judge']]:<20} & {r['icc2']:.3f} & "
            f"{r['icc3']:.3f} & {r['rho']:.3f} & {r['kappa']:.3f} & "
            f"${r['mean_diff']:+.3f}$".replace("+", "+")
            + r" \\"
        )

    body = (
        "\\begin{tabular}{lccccc}\n"
        "  \\toprule\n"
        "  \\textbf{Judge} & \\textbf{ICC(2,1)} & \\textbf{ICC(3,1)} & "
        "$\\rho$ & $\\kappa$ & \\textbf{mean\\_diff} \\\\\n"
        "  \\midrule\n" + "\n".join(rows) + "\n"
        "  \\bottomrule\n"
        "\\end{tabular}\n"
    )
    _write_or_print("table_human_llm_comparison", body, write)


# ---------------------------------------------------------------------------
# 2. Worked-example verdict tables (supp.tex ~line 403 and Example 2)
# ---------------------------------------------------------------------------


def generate_worked_example_table(
    paper_id: str, material_name: str, extractor: str, write: bool = False
) -> None:
    ann_dir = ANNOTATIONS_DIR / paper_id
    result = json.loads((ann_dir / "result.json").read_text())
    human = json.loads((ann_dir / "result_human.json").read_text())

    extractor_order = human["extractor_order"]
    extractor_idx = extractor_order.index(extractor)
    human_mat = next(
        m for m in human["materials"] if m["material_name"] == material_name
    )
    human_eval = human_mat["evaluations"][extractor_idx]["evaluation"]

    llm_entry = next(e for e in result if e["synth_llm"] == extractor)
    mat_entry = next(
        m for m in llm_entry["materials"] if m["material"] == material_name
    )

    verdicts = {"Human": human_eval}
    for jev in mat_entry["evaluations"]:
        verdicts[MODEL_DISPLAY[jev["judge_llm"]]] = jev["evaluation"]

    rows = []
    for name, ev in verdicts.items():
        rows.append(
            f"    {name:<20} & {ev['scores']['overall_score']:.2f} \\\\"
        )

    body = (
        "\\begin{tabular}{lc}\n"
        "  \\toprule\n"
        "  \\textbf{Judge} & \\textbf{Overall score} \\\\\n"
        "  \\midrule\n" + "\n".join(rows) + "\n"
        "  \\bottomrule\n"
        "\\end{tabular}\n"
    )
    slug = paper_id.replace(".", "_").replace("/", "_")
    _write_or_print(f"table_example_{slug}", body, write)


# ---------------------------------------------------------------------------
# 3. tab:thermocat-vlm-extraction (supp.tex ~line 592)
# ---------------------------------------------------------------------------


def _fuzzy_detection_row(vlm: str) -> dict:
    """Plain string-matcher (no LLM judge) accuracy + detection metrics.

    Mirrors eval_vlm.evaluate(judge=None) + the same TP/FP/FN aggregation as
    results_notebook.ipynb's detection_metrics(): TP=matched series,
    FN=GT series never matched, FP=LLM series with no GT counterpart.
    """
    import eval_vlm  # local import: adds case_study_thermocatalysis to sys.path

    rows = asyncio.run(
        eval_vlm.evaluate_async(
            CATALYSIS_RESULTS_DIR / vlm,
            CATALYSIS_GT_DIR,
            metric="rmse",
            judge=None,
        )
    )
    scored = [r for r in rows if r["score"] is not None]
    mean_rmse = (
        sum(r["score"] for r in scored) / len(scored)
        if scored
        else float("nan")
    )

    tp = fp = fn = 0
    for r in rows:
        if r["material_llm"] is None:
            fn += r["n_gt_series"] or 0
            continue
        gt_n = r["n_gt_series"] or 0
        llm_n = r["n_llm_series"] or 0
        matched = r["n_matched_series"] or 0
        tp += matched
        fn += gt_n - matched
        fp += llm_n - matched

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    return {
        "vlm": vlm,
        "mean_rmse": mean_rmse,
        "n_scored": len(scored),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _llm_matched_detection_row(vlm: str) -> dict:
    """Paraphrase-tolerant (LLM-judge) matching, from the pre-computed CSVs.

    Same source + aggregation as results_notebook.ipynb sections 4-5.
    """
    df = pd.read_csv(CATALYSIS_LLM_MATCH_DIR / f"{vlm}_llm_match.csv")
    scored = df[df["rmse"].notna()]
    mean_rmse = scored["rmse"].mean() if len(scored) else float("nan")

    def _to_int(v) -> int:
        return 0 if pd.isna(v) else int(v)

    tp = fp = fn = 0
    for _, r in df.iterrows():
        if pd.isna(r["material_llm"]):
            fn += _to_int(r["n_gt_series"])
            continue
        gt_n = _to_int(r["n_gt_series"])
        llm_n = _to_int(r["n_llm_series"])
        matched = _to_int(r["n_matched_series"])
        tp += matched
        fn += gt_n - matched
        fp += llm_n - matched

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    return {
        "vlm": vlm,
        "mean_rmse": mean_rmse,
        "n_scored": len(scored),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def generate_thermocat_vlm_table(write: bool = False) -> None:
    # deepseek-v3.2 is excluded: its OpenRouter endpoint has no image input
    # support (confirmed by 404s in data/results_catalysis_string_match/
    # deepseek-v3.2.log; only 3/26 papers completed). It is nonetheless the
    # *name-matching judge* used to produce the "LLM-matched" column below --
    # two different roles, both worth stating explicitly in the caption.
    vlms = ["qwen3.5-397b-a17b", "claude-sonnet-4.6", "gemini-3-flash"]

    rows = []
    for vlm in vlms:
        fuzzy = _fuzzy_detection_row(vlm)
        llm_matched = _llm_matched_detection_row(vlm)
        rows.append(
            f"\\multirow{{2}}{{*}}{{{MODEL_DISPLAY[vlm]}}}\n"
            f"  & Fuzzy      & {fuzzy['mean_rmse']:.3f} & "
            f"{fuzzy['n_scored']} & "
            f"{fuzzy['tp']} & {fuzzy['fp']} & {fuzzy['fn']} & "
            f"{fuzzy['precision']:.3f} & "
            f"{fuzzy['recall']:.3f} / {fuzzy['f1']:.3f} \\\\\n"
            f"  & LLM-matched & {llm_matched['mean_rmse']:.3f} & "
            f"{llm_matched['n_scored']} & {llm_matched['tp']} & "
            f"{llm_matched['fp']} & "
            f"{llm_matched['fn']} & {llm_matched['precision']:.3f} & "
            f"{llm_matched['recall']:.3f} / "
            f"\\textbf{{{llm_matched['f1']:.3f}}} \\\\"
        )

    body = (
        "\\begin{tabular}{llccccccc}\n"
        "\\toprule\n"
        "VLM & Matching & Mean RMSE & $n_{\\text{scored}}$ & TP & FP & FN & "
        "Precision & Recall / F1 \\\\\n"
        "\\midrule\n" + "\n".join(rows) + "\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
    )
    _write_or_print("table_thermocat_vlm_extraction", body, write)


# ---------------------------------------------------------------------------
# 4. table:llm-syn-scores-synthesis-type / -material-type (supp.tex ~line 740)
# ---------------------------------------------------------------------------

SCORE_CRITERIA = [
    "structural_completeness_score",
    "material_extraction_score",
    "process_steps_score",
    "equipment_extraction_score",
    "conditions_extraction_score",
    "semantic_accuracy_score",
    "format_compliance_score",
    "overall_score",
]


def _load_full_synth_df() -> pd.DataFrame:
    """Loads LeMaterial/LeMat-Synth "full" (all 3 sources), one row per
    procedure, with the 8 judge score columns flattened out of `evaluation`.
    """
    from datasets import load_dataset

    ds = load_dataset("LeMaterial/LeMat-Synth", "full")
    dfs = [d.to_pandas() for d in ds.values()]
    df = pd.concat(dfs, ignore_index=True)
    df = df[df["evaluation"].notna()].copy()
    for crit in SCORE_CRITERIA:
        df[crit] = df["evaluation"].apply(
            lambda e, crit=crit: (e or {}).get("scores", {}).get(crit)
        )
    return df


def _score_summary_table(
    df: pd.DataFrame, group_col: str, row_label: str
) -> str:
    grouped = df.groupby(group_col)
    counts = grouped.size().sort_values(ascending=False)

    rows = []
    for group_val, count in counts.items():
        sub = grouped.get_group(group_val)
        cells = []
        for crit in SCORE_CRITERIA:
            vals = sub[crit].dropna()
            if len(vals) == 0:
                cells.append("-")
            elif len(vals) == 1:
                cells.append(f"{vals.iloc[0]:.2f}$\\pm$nan")
            else:
                cells.append(f"{vals.mean():.2f}$\\pm${vals.std():.2f}")
        label = str(group_val).replace("&", "\\&")
        rows.append(f"{label} & " + " & ".join(cells) + f" & {count} \\\\")

    header = (
        f"\\textbf{{{row_label}}} & \\textbf{{Structural}} & "
        "\\textbf{Material} & "
        "\\textbf{Process} & \\textbf{Equipment} & \\textbf{Condition} & "
        "\\textbf{Semantic} & \\textbf{Format} & \\textbf{Overall} & "
        "\\textbf{Count} \\\\ \n"
        "\\textbf{} & \\textbf{completeness} & \\textbf{completeness} & "
        "\\textbf{steps} & \\textbf{extraction} & \\textbf{extraction} & "
        "\\textbf{accuracy} & \\textbf{compliance} & \\textbf{score} & "
        "\\textbf{} \\\\"
    )
    return (
        "\\begin{tabular}{lccccccccc}\n"
        "\\toprule\n" + header + "\n"
        "\\midrule\n" + "\n".join(rows) + "\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
    )


def generate_synthesis_method_table(write: bool = False) -> None:
    df = _load_full_synth_df()
    body = _score_summary_table(df, "synthesis_method", "Synthesis\\\\method")
    _write_or_print("table_llm_syn_scores_synthesis_type", body, write)


def generate_material_category_table(write: bool = False) -> None:
    df = _load_full_synth_df()
    body = _score_summary_table(df, "material_category", "Material\\\\category")
    _write_or_print("table_llm_syn_scores_material_type", body, write)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write",
        action="store_true",
        help=f"Write .tex fragments to {TABLES_OUT_DIR} instead of stdout.",
    )
    args = parser.parse_args()

    generate_judge_agreement_table(write=args.write)
    generate_worked_example_table(
        "cond-mat.0602418",
        "Bi1.74Pb0.38Sr1.88CuO6+\u03b4",
        "gemini-3-flash",
        write=args.write,
    )
    generate_worked_example_table(
        "64b40972b605c6803bd37ab4",
        "WFe2Ni-red",
        "deepseek-v3.2",
        write=args.write,
    )
    generate_worked_example_table(
        "1605.04038",
        "(La0.3Sr0.7)(Al0.65Ta0.35)O3/SrTiO3",
        "gemini-3-flash",
        write=args.write,
    )
    generate_worked_example_table(
        "1706.00484",
        "SrTiO3",
        "gemini-3-flash",
        write=args.write,
    )
    generate_worked_example_table(
        "cond-mat.0503432",
        "GdCo2",
        "gemini-3-flash",
        write=args.write,
    )
    generate_worked_example_table(
        "1902.03049",
        "5-AGNR",
        "gemini-3-flash",
        write=args.write,
    )
    generate_thermocat_vlm_table(write=args.write)
    generate_synthesis_method_table(write=args.write)
    generate_material_category_table(write=args.write)


if __name__ == "__main__":
    main()
