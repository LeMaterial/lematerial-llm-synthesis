from abc import ABC, abstractmethod


class BasePDFExtractor(ABC):
    @abstractmethod
    def extract_to_markdown_with_figures_embedded(pdf_data: bytes) -> str:
        """
        Extracts text and figures from a PDF and returns them as markdown with embedded figures.

        Args:
            pdf_data: The PDF data as bytes.

        Returns:
            The extracted text as markdown with embedded figures.
        """
        pass
