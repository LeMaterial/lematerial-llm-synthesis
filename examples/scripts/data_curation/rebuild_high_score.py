"""Rebuild LeMaterial/LeMat-Synth's "high_score" config from "full".

Replaces the old filter (evaluation.scores.overall_score > 4) with a
stricter-on-failure-modes, looser-on-score filter, chosen by inspecting the
joint distribution of judge overall_score and number of extracted synthesis
steps: overall_score > 4 alone let through degenerate zero/one-step
extractions with an inflated score (~23.6% of the old high_score subset had
<=1 step). The new filter:

    evaluation.scores.overall_score > 3.25
    AND len(structured_synthesis.steps) > 1

keeps ~30.2k / 58.3k rows (51.8%) with zero rows at <=1 step, vs. 16.4k rows
(28.2%) and 23.6% failure-mode contamination under the old filter.

Opens a PR against LeMaterial/LeMat-Synth for review (create_pr=True) rather
than pushing directly.
"""

import argparse

from datasets import DatasetDict, load_dataset

REPO_ID = "LeMaterial/LeMat-Synth"
SCORE_THRESHOLD = 3.25
MIN_STEPS = 1  # keep len(steps) > MIN_STEPS


def keep_row(example: dict) -> bool:
    ev = example["evaluation"]
    synth = example["structured_synthesis"]
    if ev is None or synth is None:
        return False
    score = ev["scores"]["overall_score"]
    steps = synth.get("steps") or []
    return score > SCORE_THRESHOLD and len(steps) > MIN_STEPS


def main(push: bool) -> None:
    full = load_dataset(REPO_ID, "full")

    filtered = {}
    for split, ds in full.items():
        subset = ds.filter(keep_row)
        filtered[split] = subset
        print(
            f"high_score/{split}: {subset.num_rows} / {ds.num_rows} rows kept"
        )

    total_before = sum(ds.num_rows for ds in full.values())
    total_after = sum(ds.num_rows for ds in filtered.values())
    print(
        f"\nTOTAL: {total_after} / {total_before} rows kept "
        f"({100 * total_after / total_before:.1f}%)"
    )

    if push:
        DatasetDict(filtered).push_to_hub(
            REPO_ID,
            config_name="high_score",
            create_pr=True,
            commit_message=(
                "Rebuild high_score subset: overall_score>3.25 & "
                "len(steps)>1 (replaces overall_score>4)"
            ),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--push", action="store_true")
    args = parser.parse_args()
    main(push=args.push)
