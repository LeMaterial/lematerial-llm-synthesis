"""Regenerate insights_judge_ranking_loo.csv with LLM-assisted name matching.

Only insights_judge_ranking_loo.csv depends on cross-matching human material
names to LLM-extracted material names (via compare_multi_llm_results_complete
.load_annotations); every other insights_*.csv (judge_extractor_matrix,
self_preference, judge_behavior, ...) is built from analyze_judge_extractor_
insights.load_long, which keys rows by synth_llm identity, not name-matching,
so those are unaffected and not regenerated here.

Writes to a new directory (results/agreement_analysis_llm_match/) rather than
overwriting the original, so the string-matcher-only CSVs -- and the
manuscript numbers already cited from them -- stay available as a fallback.

Usage:
    uv run python \\
        examples/scripts/evaluation/regenerate_judge_ranking_llm_match.py
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

_repo_root = Path(__file__).resolve().parents[3]
load_dotenv(_repo_root / ".env", override=True)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_judge_extractor_insights import (  # noqa: E402
    SKIP_FOLDERS,
    judge_ranking_variants,
)
from compare_multi_llm_results_llm_match import (  # noqa: E402
    build_name_matcher_judge,
    load_annotations,
)

OUTPUT_DIR = "results/agreement_analysis_llm_match"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations-dir", default="annotations")
    parser.add_argument(
        "--matcher-model",
        default="claude-sonnet-4.6",
        help="LLM_REGISTRY model name for the name-matcher judge",
    )
    parser.add_argument(
        "--no-llm-match",
        action="store_true",
        help="Disable the LLM fallback (string-matcher only, for A/B diffing)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(message)s",
    )
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    matcher = (
        None
        if args.no_llm_match
        else build_name_matcher_judge(args.matcher_model)
    )
    human_df, llm_df = load_annotations(
        args.annotations_dir, SKIP_FOLDERS, matcher=matcher
    )

    df = judge_ranking_variants(human_df, llm_df)
    path = os.path.join(OUTPUT_DIR, "insights_judge_ranking_loo.csv")
    df.to_csv(path, index=False)
    print("\n===== insights_judge_ranking_loo.csv (LLM-matched) =====")
    print(df.to_string(index=False))
    print(f"\nWritten to {path}")


if __name__ == "__main__":
    main()
