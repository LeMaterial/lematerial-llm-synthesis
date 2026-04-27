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

from llm_synthesis.utils.paper_id_utils import (
    folder_id_to_hf_id,
    hf_id_to_folder_id,
)

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

    needed_hf = {folder_id_to_hf_id(pid) for pid in paper_ids}
    collected = {}
    try:
        dataset = load_dataset(dataset_id, split=split, streaming=True)
        for row in dataset:
            pid_hf = row.get("id")
            if pid_hf in needed_hf:
                folder_id = hf_id_to_folder_id(pid_hf)
                row_dict = dict(row)
                row_dict["id"] = folder_id
                collected[folder_id] = row_dict
                if len(collected) == len(needed_hf):
                    break
    except (TypeError, ValueError):
        # Dataset may not support streaming; fall back to full load + filter
        dataset = load_dataset(dataset_id, split=split)
        id_to_idx = {row["id"]: i for i, row in enumerate(dataset)}
        found_ids = [p for p in paper_ids if folder_id_to_hf_id(p) in id_to_idx]
        missing = [p for p in paper_ids if p not in found_ids]
        if missing:
            print(f"Missing IDs (not in {dataset_id} {split}): {missing}")
        indices = [id_to_idx[folder_id_to_hf_id(pid)] for pid in found_ids]
        subset = dataset.select(indices)
        # Normalize id column to folder id (cond-mat.XX) for consistency
        def _norm_id(row, idx):
            row = dict(row)
            row["id"] = found_ids[idx]
            return row

        subset = Dataset.from_list(
            [_norm_id(subset[i], i) for i in range(len(subset))]
        )
        return found_ids, subset

    missing = [p for p in paper_ids if p not in collected]
    if missing:
        print(f"Missing IDs (not in {dataset_id} {split}): {missing}")

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
