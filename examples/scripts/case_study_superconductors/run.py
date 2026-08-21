#!/usr/bin/env python3
"""
Batch runner for the superconductors case study.

Extracts synthesis procedures, critical temperatures (Tc) from text, and
Tc from R(T) plots via geometric construction.

Results are saved per paper (JSON) and aggregated into a growing
tc_master.csv across all processed papers.

Usage:
    python run.py /path/to/pdf_folder
    python run.py /path/to/pdf_folder /path/to/output
    python run.py /path/to/pdf_folder --max 5
    python run.py /path/to/pdf_folder --skip-existing
    python run.py /path/to/pdf_folder --skip-figures
"""

import argparse
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

from llm_synthesis.config.domain_config import DomainConfig
from llm_synthesis.runners.batch_runner import BatchRunner

# Add src to path so the package is importable without installation
_SRC = Path(__file__).resolve().parents[3] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

load_dotenv(Path(__file__).resolve().parents[3] / ".env", override=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

DEFAULT_PDF_DIR = "../data/pdf_papers"
DEFAULT_OUTPUT_DIR = "../data/results_superconductors"

GEMINI_MODEL = "gemini-3.0-flash"
CLAUDE_MODEL = "claude-sonnet-4-20250514"
MATERIAL_MODEL = "gemini-3.0-pro"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch Tc extraction pipeline for superconductors.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "pdf_dir",
        nargs="?",
        default=DEFAULT_PDF_DIR,
        help=f"Directory containing PDF files (default: {DEFAULT_PDF_DIR})",
    )
    parser.add_argument(
        "output_dir",
        nargs="?",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--max", type=int, default=None, help="Max papers to process (testing)"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip papers that already have results",
    )
    parser.add_argument(
        "--skip-figures",
        action="store_true",
        help="Skip figure extraction and VLM Tc (text-only mode)",
    )
    args = parser.parse_args()

    runner = BatchRunner(
        domain_config=DomainConfig.for_superconductivity(
            claude_model=CLAUDE_MODEL
        ),
        gemini_model=GEMINI_MODEL,
        claude_model=CLAUDE_MODEL,
        material_model=MATERIAL_MODEL,
        linker_model=GEMINI_MODEL,
        synthesis_max_tokens=32_000,
        linker_max_tokens=16_000,
    )
    runner.run(
        pdf_dir=args.pdf_dir,
        output_dir=args.output_dir,
        max_papers=args.max,
        skip_existing=args.skip_existing,
        skip_figures=args.skip_figures,
    )


if __name__ == "__main__":
    main()
