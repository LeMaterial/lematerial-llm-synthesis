#!/usr/bin/env python3
"""Download arxiv PDFs for the superconductor_keywords_and_LLM split.

Usage:
    python download_pdfs.py --max 5
    python download_pdfs.py --out ../../../data/pdf_papers_superconductors
"""

import argparse
import time
from pathlib import Path

import httpx
from datasets import load_dataset

DATASET_PATH = (
    "hf://datasets/LeMaterial/LeMat-Synth-Papers/"
    "superconductor_keywords_and_LLM/full-00000-of-00001.parquet"
)
DEFAULT_OUT = "../../../data/pdf_papers_superconductors"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--max", type=int, default=None)
    parser.add_argument(
        "--ids",
        nargs="*",
        default=None,
        help="Specific HF dataset ids to fetch",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help=(
            "Randomly sample N papers instead of taking the first N -- "
            "dataset rows are roughly chronological (arxiv id order), so "
            "--max alone clusters on whatever years happen to be first."
        ),
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed for --sample"
    )
    args = parser.parse_args()

    out_dir = Path(__file__).resolve().parent / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(
        "parquet", data_files=DATASET_PATH, split="train"
    ).to_pandas()
    if args.sample:
        # Shuffle the WHOLE dataset once with a fixed seed, then take the
        # first N -- re-running with a larger --sample extends the same
        # queue (first K already-downloaded stay first, only the new tail
        # is fresh) instead of re-picking an unrelated random N each time.
        df = df.sample(frac=1, random_state=args.seed).head(args.sample)
    elif args.max:
        df = df.head(args.max)
    if args.ids:
        wanted = set(args.ids)
        df = df[df["id"].isin(wanted)]

    with httpx.Client(follow_redirects=True, timeout=60) as client:
        for i, row in df.iterrows():
            # old-style arxiv ids like "cond-mat/0102313" have a "/" -- sanitize
            # for the filesystem, matching run_from_hf.py's lookup convention.
            safe_id = row["id"].replace("/", "_")
            dest = out_dir / f"{safe_id}.pdf"
            if dest.exists():
                print(f"  [{i + 1}/{len(df)}] skip (exists): {row['id']}")
                continue
            print(
                f"  [{i + 1}/{len(df)}] downl. {row['id']} <- {row['pdf_url']}"
            )
            try:
                resp = client.get(row["pdf_url"])
                resp.raise_for_status()
                dest.write_bytes(resp.content)
            except Exception as e:
                print(f"    FAILED: {e}")
            time.sleep(
                0.5
            )  # ponytail: arxiv rate limit courtesy, raise if flaky

    print(f"Done. PDFs in {out_dir}")


if __name__ == "__main__":
    main()
