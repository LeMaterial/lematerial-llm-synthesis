from abc import abstractmethod

from llm_synthesis.models.figure import FigureInfo
from llm_synthesis.transformers.base import ExtractorInterface


class FigureExtractorInterface(ExtractorInterface[str, list[FigureInfo]]):
    """
    Interface for extracting figures from a markdown text. Implementations should return a list of FigureInfo objects found in the given markdown text.
    """

    @abstractmethod
    def extract(self, input: str) -> list[FigureInfo]:
        """
        Extract figures from the given markdown text.

        Args:
            input (str): The markdown text from which to extract figures.

        Returns:
            list[FigureInfo]: A list of extracted figure information objects.
        """
        pass
