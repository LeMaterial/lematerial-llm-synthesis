from enum import Enum

from llm_synthesis.services.pdf_extraction.base_pdf_extractor import BasePDFExtractor
from llm_synthesis.services.pdf_extraction.docling_pdf_extractor import (
    DoclingPDFExtractor,
)
from llm_synthesis.services.pdf_extraction.mistral_pdf_extractor import (
    MistralPDFExtractor,
)


class PDFExtractorEnum(Enum):
    """Enum for the pdf extractors."""

    DOCLING = "docling"
    MISTRAL = "mistral"

    def __str__(self) -> str:
        return self.value


def create_pdf_extractor(pdf_extractor: PDFExtractorEnum) -> BasePDFExtractor:
    """
    Factory function to create a PDF extractor based on the settings.

    Returns:
        A PDF extractor instance.

    """
    if pdf_extractor == PDFExtractorEnum.DOCLING:
        return DoclingPDFExtractor()
    elif pdf_extractor == PDFExtractorEnum.MISTRAL:
        return MistralPDFExtractor()
    else:
        raise ValueError(f"Unsupported PDF extractor type: {pdf_extractor}")
