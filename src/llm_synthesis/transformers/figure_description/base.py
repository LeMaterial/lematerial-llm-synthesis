from abc import abstractmethod

from llm_synthesis.models.figure import FigureInfoWithPaper
from llm_synthesis.transformers.base import ExtractorInterface


class FigureDescriptionExtractorInterface(ExtractorInterface[FigureInfoWithPaper, str]):
    """
    Interface for a figure description extractor that takes a figure info with paper and returns a figure description.
    """

    @abstractmethod
    def extract(self, input: FigureInfoWithPaper) -> str:
        """
        Extract a description for the given figure info with paper context.

        Args:
            input (FigureInfoWithPaper): The figure information along with its associated paper.

        Returns:
            str: The generated description for the figure.
        """
        pass
