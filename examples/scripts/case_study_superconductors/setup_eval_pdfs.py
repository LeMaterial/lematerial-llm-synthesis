"""Create a folder with symlinks/copies of just the 18 ground-truth PDFs.

Usage:
    python setup_eval_pdfs.py <path_to_selected_papers_dir> [--output <output_dir>]

Example:
    python setup_eval_pdfs.py /path/to/selected_papers --output /tmp/tc_eval_pdfs
"""

import argparse
import shutil
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
GT_PATH = SCRIPT_DIR / "ground_truth_tc.xlsx"


def main():
    parser = argparse.ArgumentParser(description="Set up eval PDF folder")
    parser.add_argument("pdf_dir", type=Path, help="Path to selected_papers dir with all PDFs")
    parser.add_argument("--output", type=Path, default=SCRIPT_DIR / "eval_pdfs",
                        help="Output folder for filtered PDFs (default: eval_pdfs/)")
    args = parser.parse_args()

    gt = pd.read_excel(GT_PATH)
    paper_ids = gt["paper_id"].unique()

    args.output.mkdir(parents=True, exist_ok=True)

    found, missing = 0, []
    for pid in paper_ids:
        src = args.pdf_dir / f"{pid}.pdf"
        dst = args.output / f"{pid}.pdf"
        if src.exists():
            shutil.copy2(src, dst)
            found += 1
        else:
            missing.append(pid)

    print(f"Copied {found} PDFs to {args.output}")
    if missing:
        print(f"Missing: {missing}")


if __name__ == "__main__":
    main()
