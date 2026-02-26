#!/usr/bin/env python3
"""
Load paper rows from LeMat-Synth-Papers (sample_for_evaluation) for paper IDs
that correspond to annotation folder names under annotations/.

Directory names under annotations/ match the id column in:
https://huggingface.co/datasets/LeMaterial/LeMat-Synth-Papers

Use the `text_paper` column from the returned rows as the source for recipe
extraction and judge input.
"""

import argparse
import json
from pathlib import Path

from datasets import Dataset, load_dataset

DATASET_ID = "LeMaterial/LeMat-Synth-Papers"
SPLIT = "sample_for_evaluation"


def get_annotation_paper_ids(annotations_dir: Path) -> list[str]:
    """Return paper IDs from subdirectory names in annotations_dir."""
    if not annotations_dir.is_dir():
        return []
    return sorted(
        d.name
        for d in annotations_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    )


def load_papers_for_annotation_ids(
    annotations_dir: Path,
    *,
    dataset_id: str = DATASET_ID,
    split: str = SPLIT,
):
    """
    Load dataset rows for paper IDs that have annotation folders.

    Uses streaming so only matching rows are kept in memory; stops reading
    once all requested IDs are found.

    Returns (paper_ids, dataset_subset).
    dataset_subset is a HuggingFace Dataset with only rows whose id is in
    paper_ids.
    """
    paper_ids = get_annotation_paper_ids(annotations_dir)
    if not paper_ids:
        return paper_ids, None

    needed = set(paper_ids)
    collected = {}
    try:
        dataset = load_dataset(dataset_id, split=split, streaming=True)
        for row in dataset:
            pid = row.get("id")
            if pid in needed:
                collected[pid] = dict(row)
                if len(collected) == len(needed):
                    break
    except (TypeError, ValueError):
        # Dataset may not support streaming; fall back to full load + filter
        dataset = load_dataset(dataset_id, split=split)
        id_to_idx = {row["id"]: i for i, row in enumerate(dataset)}
        missing = [p for p in paper_ids if p not in id_to_idx]
        if missing:
            print(f"Missing IDs (not in {dataset_id} {split}): {missing}")
            # raise ValueError(
            #     f"Paper IDs not found in {dataset_id} ({split}): {missing}. "
            #     "Annotation folder names must match the dataset id column."
            # ) from None
        found_ids = [p for p in paper_ids if p in id_to_idx]
        subset = dataset.select([id_to_idx[pid] for pid in found_ids])
        return found_ids, subset

    missing = [p for p in paper_ids if p not in collected]
    if missing:
        print(f"Missing IDs (not in {dataset_id} {split}): {missing}")
        # raise ValueError(
        #     f"Paper IDs not found in {dataset_id} ({split}): {missing}. "
        #     "Annotation folder names must match the dataset id column."
        # )

    found_ids = [p for p in paper_ids if p in collected]
    subset = Dataset.from_list([collected[pid] for pid in found_ids])
    return found_ids, subset


def main():
    parser = argparse.ArgumentParser(
        description="Load LeMat-Synth-Papers rows for annotation folder IDs"
    )
    parser.add_argument(
        "--annotations-dir",
        type=Path,
        default=Path("annotations"),
        help="Root dir with one subfolder per paper (default: annotations)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=DATASET_ID,
        help=f"HuggingFace dataset ID (default: {DATASET_ID})",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=SPLIT,
        help=f"Dataset split (default: {SPLIT})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Save path: .parquet, .json, .jsonl, or dir for HF dataset.",
    )
    parser.add_argument(
        "--list-ids",
        action="store_true",
        help="Only print paper IDs (one per line) and exit.",
    )
    args = parser.parse_args()

    annotations_dir = args.annotations_dir.resolve()
    paper_ids, subset = load_papers_for_annotation_ids(
        annotations_dir, dataset_id=args.dataset, split=args.split
    )

    if args.list_ids:
        for pid in paper_ids:
            print(pid)
        return

    if subset is None:
        print("No annotation folders found.")
        return

    n, m = len(paper_ids), len(subset)
    print(f"Found {n} annotation folder(s). Loaded {m} row(s).")

    if args.output:
        out = args.output.resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.suffix == ".parquet":
            subset.to_parquet(out)
        elif out.suffix == ".json":
            rows = list(subset)
            with open(out, "w", encoding="utf-8") as f:
                json.dump(rows, f, indent=2, ensure_ascii=False, default=str)
        elif out.suffix == ".jsonl":
            subset.to_json(out)
        else:
            subset.save_to_disk(str(out))
        print(f"Saved to {out}")


if __name__ == "__main__":
    main()
