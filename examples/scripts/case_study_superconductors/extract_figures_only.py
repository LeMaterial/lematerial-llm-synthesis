#!/usr/bin/env python3
"""Run just OCR + Florence-2 figure extraction for a folder of PDFs,
caching results to <output_dir>/_figure_cache/<paper_id>/figures.json --
same cache format run_from_hf.py reads. Lets the slow CPU-bound step run
ahead of time, independent of which VLM/prompt is used later.

Usage:
    python extract_figures_only.py /path/to/pdf_dir /path/to/output_dir \
         --max 100
"""

import argparse
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

_SRC = Path(__file__).resolve().parents[3] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(Path(__file__).resolve().parent))

load_dotenv(Path(__file__).resolve().parents[3] / ".env", override=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf_dir")
    parser.add_argument("output_dir")
    parser.add_argument("--max", type=int, default=None)
    args = parser.parse_args()

    from run_from_hf import build_pipeline, load_or_extract_figures

    from llm_synthesis.transformers.pdf_extraction import MistralPDFExtractor

    pdf_dir = Path(args.pdf_dir)
    output_dir = Path(args.output_dir)
    cache_dir = output_dir / "_figure_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    pdf_paths = sorted(pdf_dir.glob("*.pdf"))
    if args.max:
        pdf_paths = pdf_paths[: args.max]

    pdf_extractor = MistralPDFExtractor(structured=False)
    pipeline = build_pipeline()

    done = 0
    for pdf_path in pdf_paths:
        cache_file = cache_dir / pdf_path.stem / "figures.json"
        if cache_file.exists():
            logger.info(
                "[%d/%d] cached: %s", done + 1, len(pdf_paths), pdf_path.stem
            )
            done += 1
            continue
        try:
            figures = load_or_extract_figures(
                pipeline, pdf_extractor, pdf_path, cache_dir
            )
            logger.info(
                "[%d/%d] %s: %d figures",
                done + 1,
                len(pdf_paths),
                pdf_path.stem,
                len(figures),
            )
        except Exception as e:
            logger.error(
                "[%d/%d] FAILED %s: %s",
                done + 1,
                len(pdf_paths),
                pdf_path.stem,
                e,
            )
        done += 1

    logger.info("Done. %d papers, cache at %s", done, cache_dir)


if __name__ == "__main__":
    main()
