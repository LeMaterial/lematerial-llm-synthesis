"""Pipeline for PDF conversion and figure extraction only."""

import logging

from llm_synthesis.services.pipelines.synthesis_performance_pipeline import (
    SynthesisPerformancePipeline,
)
from llm_synthesis.transformers.pdf_extraction.mistral_pdf_extractor import (
    MistralPDFExtractor,
)

logger = logging.getLogger(__name__)


class PlotExtractionPipeline(SynthesisPerformancePipeline):
    """Lightweight pipeline for PDF conversion and figure extraction.

    Inherits from SynthesisPerformancePipeline to reuse extract_figures().
    Adds PDF-to-markdown conversion methods.
    """

    def __init__(self):
        """Initialize with only plot-related components, skipping parent __init__."""
        self.pdf_extractor = MistralPDFExtractor()
        self.plot_extractor = None  # VLM plot extractor, not needed here
        self.plot_filter = None

    def convert_pdf_from_bytes(self, pdf_bytes: bytes) -> str:
        """Convert PDF bytes to markdown with embedded base64 images."""
        return self.pdf_extractor.forward(pdf_bytes)

    def convert_pdf_from_url(self, pdf_url: str) -> str:
        """Download PDF from URL and convert to markdown."""
        import requests

        response = requests.get(pdf_url, timeout=60)
        response.raise_for_status()
        return self.convert_pdf_from_bytes(response.content)

