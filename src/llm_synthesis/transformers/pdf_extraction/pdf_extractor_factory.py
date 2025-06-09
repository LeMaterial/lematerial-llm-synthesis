from enum import Enum

from llm_synthesis.transformers.pdf_extraction.base import (
    PdfExtractorInterface,
)
from llm_synthesis.transformers.pdf_extraction.docling_pdf_extractor import (
    DoclingPDFExtractor,
)
from llm_synthesis.transformers.pdf_extraction.mistral_pdf_extractor import (
    MistralPDFExtractor,
)


class PDFExtractorEnum(Enum):
    """Enum for the pdf extractors."""

    DOCLING = "docling"
    MISTRAL = "mistral"

    def __str__(self) -> str:
        return self.value


def create_pdf_extractor(
    extractor_type: PDFExtractorEnum,
) -> PdfExtractorInterface:
    if extractor_type == PDFExtractorEnum.DOCLING:
        return DoclingPDFExtractor()
    elif extractor_type == PDFExtractorEnum.MISTRAL:
        return MistralPDFExtractor()
    else:
        raise ValueError(f"Invalid extractor type: {extractor_type}")
