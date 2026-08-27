"""Link ground_truth_tc.xlsx (human) rows to VLM tc_master.csv rows.

Same two-stage approach as
examples/scripts/evaluation/compare_multi_llm_results_llm_match.py:
string match first (eval_utils.find_best_matches, threshold 0.7), then
DspyNameMatcherJudge as a fallback for materials the string matcher misses
within the same paper (handles stoichiometry-notation differences, e.g.
"Ca1-xLaxFe2As2" vs "Ca0.8La0.2Fe2As2").

Fixes the n=10 calibration-plot bug in results_supercond.ipynb, which joined
against a stale CSV (results/results_superconductors/tc_master_snippet.csv)
instead of the correct manual annotations (ground_truth_tc.xlsx), using exact
string equality only.

Usage:
    uv run python link_tc_ground_truth.py [--vlm PATH_TO_tc_master.csv]

Outputs linked_tc.csv: paper_id, gt_material, vlm_material, tc_human,
tc_vlm, match_method.
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
_EVAL_DIR = str(_SCRIPT_DIR.parents[1] / "scripts" / "evaluation")
if _EVAL_DIR not in sys.path:
    sys.path.insert(0, _EVAL_DIR)
from eval_utils import find_best_matches  # noqa: E402

_SRC_DIR = str(_SCRIPT_DIR.parents[2] / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
from llm_synthesis.metrics.judge.name_matcher_judge import (  # noqa: E402
    DspyNameMatcherJudge,
    build_name_match_inputs,
)
from llm_synthesis.utils.llms import LLM_REGISTRY  # noqa: E402

DEFAULT_GT_PATH = _SCRIPT_DIR / "ground_truth_tc.xlsx"
DEFAULT_VLM_PATH = Path(
    "/mnt/home/magled/lematerial-llm-synthesis/data/"
    "results_superconductors_hf_snippet19/tc_master.csv"
)


def build_name_matcher_judge(model_name: str = "claude-sonnet-4.6"):
    """Same construction as compare_multi_llm_results_llm_match.py."""
    import dspy

    cfg = LLM_REGISTRY.configs[model_name]
    kwargs = dict(cfg.extra_kwargs or {})
    if cfg.api_key:
        kwargs["api_key"] = cfg.api_key
    if cfg.api_base:
        kwargs["api_base"] = cfg.api_base
    lm = dspy.LM(cfg.model, temperature=0.1, **kwargs)
    return DspyNameMatcherJudge(lm=lm)


def load_ground_truth(path: Path, gt_strategy: str = "text") -> pd.DataFrame:
    """gt_strategy: "text" (tc_text_human only), "plot" (tc_human_from_plot
    only), or "text_then_plot" (text where stated, else plot)."""
    gt = pd.read_excel(path)
    gt["paper_id_norm"] = gt["paper_id"].astype(str).str.split("_").str[0]
    text_tc = pd.to_numeric(gt["tc_text_human"], errors="coerce")
    plot_tc = pd.to_numeric(gt["tc_human_from_plot"], errors="coerce")
    if gt_strategy == "text":
        gt["tc_human"] = text_tc
    elif gt_strategy == "plot":
        gt["tc_human"] = plot_tc
    elif gt_strategy == "text_then_plot":
        gt["tc_human"] = text_tc.fillna(plot_tc)
    else:
        raise ValueError(f"Unknown gt_strategy: {gt_strategy!r}")
    return gt


def load_vlm(path: Path) -> pd.DataFrame:
    vlm = pd.read_csv(path)
    vlm["paper_id_norm"] = (
        vlm["paper_id"].astype(str).str.replace("cond-mat_", "", regex=False)
    )
    return vlm


def _llm_match_unmatched(matcher, paper_id, unmatched_gt_names, vlm_candidates):
    if matcher is None or not unmatched_gt_names or not vlm_candidates:
        return {}
    result = matcher.forward(
        build_name_match_inputs(
            unmatched_gt_names, vlm_candidates, context=f"paper_id={paper_id}"
        )
    )
    matches = {}
    for pair in result.matches:
        if pair.llm_name is None or pair.llm_name not in vlm_candidates:
            continue
        if pair.confidence == "low":
            logging.info(
                "LLM match SKIPPED (confidence=low) [%s]: %r -> %r",
                paper_id,
                pair.gt_name,
                pair.llm_name,
            )
            continue
        matches[pair.gt_name] = pair.llm_name
        logging.info(
            "LLM match [%s]: %r -> %r (confidence=%s)",
            paper_id,
            pair.gt_name,
            pair.llm_name,
            pair.confidence,
        )
    return matches


def link(gt: pd.DataFrame, vlm: pd.DataFrame, matcher=None) -> pd.DataFrame:
    rows = []
    for paper_id, gt_paper in gt.groupby("paper_id_norm"):
        vlm_paper = vlm[vlm["paper_id_norm"] == paper_id]
        if vlm_paper.empty:
            continue

        gt_names = gt_paper["material"].tolist()
        vlm_names = vlm_paper["material"].tolist()

        string_matches = find_best_matches(
            gt_names, vlm_names, similarity_threshold=0.7
        )

        used_vlm = set(string_matches.values())
        unmatched_gt = [g for g in gt_names if g not in string_matches]
        remaining_vlm = [v for v in vlm_names if v not in used_vlm]

        llm_matches = _llm_match_unmatched(
            matcher, paper_id, unmatched_gt, remaining_vlm
        )

        for _, gt_row in gt_paper.iterrows():
            gt_mat = gt_row["material"]
            vlm_mat, method = None, None
            if gt_mat in string_matches:
                vlm_mat, method = string_matches[gt_mat], "string"
            elif gt_mat in llm_matches:
                vlm_mat, method = llm_matches[gt_mat], "llm"

            tc_vlm = None
            if vlm_mat is not None:
                vlm_row = vlm_paper[vlm_paper["material"] == vlm_mat].iloc[0]
                tc_vlm = pd.to_numeric(vlm_row.get("tc_vlm"), errors="coerce")

            rows.append(
                {
                    "paper_id": paper_id,
                    "gt_material": gt_mat,
                    "vlm_material": vlm_mat,
                    "tc_human": gt_row["tc_human"],
                    "tc_vlm": tc_vlm,
                    "match_method": method,
                }
            )

    return pd.DataFrame(rows)


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(
        description="Link Tc ground truth to VLM results"
    )
    parser.add_argument("--gt", type=Path, default=DEFAULT_GT_PATH)
    parser.add_argument("--vlm", type=Path, default=DEFAULT_VLM_PATH)
    parser.add_argument("--no-llm-fallback", action="store_true")
    parser.add_argument(
        "--gt-strategy",
        choices=["text", "plot", "text_then_plot"],
        default="text",
        help=(
            "Which ground-truth Tc to use: 'text' (tc_text_human only, "
            "default), 'plot' (tc_human_from_plot only), or "
            "'text_then_plot' (text where stated, else plot)."
        ),
    )
    parser.add_argument(
        "--output", type=Path, default=_SCRIPT_DIR / "linked_tc.csv"
    )
    args = parser.parse_args()

    gt = load_ground_truth(args.gt, gt_strategy=args.gt_strategy)
    vlm = load_vlm(args.vlm)

    matcher = None if args.no_llm_fallback else build_name_matcher_judge()
    linked = link(gt, vlm, matcher=matcher)

    n_string = (linked["match_method"] == "string").sum()
    n_llm = (linked["match_method"] == "llm").sum()
    both = linked.dropna(subset=["tc_human", "tc_vlm"])

    print(f"GT rows:              {len(gt)}")
    print(f"Matched (string):      {n_string}")
    print(f"Matched (llm):         {n_llm}")
    print(f"Unmatched:             {len(linked) - n_string - n_llm}")
    print(f"Rows with both tc_human & tc_vlm: {len(both)}")

    linked.to_csv(args.output, index=False)
    print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    main()
