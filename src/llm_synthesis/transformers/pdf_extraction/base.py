from abc import abstractmethod

from llm_synthesis.transformers.base import ExtractorInterface


class PdfExtractorInterface(ExtractorInterface[bytes, str]):
    """
    Interface for a pdf extractor that takes a pdf as bytes and returns a string.
    """

    @abstractmethod
    def extract(self, input: bytes) -> str:
        pass
