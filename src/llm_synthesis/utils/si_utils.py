"""Utilities for locating and loading supplementary information (SI) files."""

from pathlib import Path

SI_PATTERNS = [
    "_SI",
    "-SI",
    "_si",
    "-si",
    "_Supporting",
    "_supporting",
    "_Supplementary",
    "_supplementary",
    "_supp",
    "_Supp",
]


def is_si_file(path: Path) -> bool:
    """Return True if the file looks like a supplementary information file."""
    return any(pattern in path.stem for pattern in SI_PATTERNS)


def find_si_file(main_paper_path: Path) -> Path | None:
    """Find the SI file that matches a main paper.

    Searches for files like MainPaper_SI.pdf, MainPaper_Supporting.pdf, etc.

    Args:
        main_paper_path: Path to the main paper file.

    Returns:
        Path to the SI file if found, None otherwise.
    """
    parent_dir = main_paper_path.parent
    main_stem = main_paper_path.stem
    for pattern in SI_PATTERNS:
        for ext in [".pdf", ".md", ".txt"]:
            si_path = parent_dir / f"{main_stem}{pattern}{ext}"
            if si_path.exists():
                return si_path
    return None


def load_file_text(path: Path, pdf_extractor=None) -> str:
    """Load text from a PDF, MD, or TXT file.

    Args:
        path: Path to the file.
        pdf_extractor: MistralPDFExtractor instance for PDF files.
                       If None, a new one is created on demand.

    Returns:
        Extracted text content.
    """
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        if pdf_extractor is None:
            from llm_synthesis.transformers.pdf_extraction import (
                MistralPDFExtractor,
            )

            pdf_extractor = MistralPDFExtractor(structured=False)
        with open(path, "rb") as f:
            return pdf_extractor.forward(f.read())
    elif suffix in [".md", ".txt"]:
        with open(path, errors="replace") as f:
            return f.read()
    else:
        raise ValueError(f"Unsupported file type: {suffix}")
