from llm_synthesis.models.figure import FigureInfo
from llm_synthesis.transformers.figure_extraction.base import (
    FigureExtractorInterface,
)
from llm_synthesis.utils.figure_utils import find_figures_in_markdown


class FigureExtractorMarkdown(FigureExtractorInterface):
    """
    Extracts figures from a markdown text using regex-based markdown parsing.
    """

    def forward(self, input: str) -> list[FigureInfo]:
        """
        Extract figures from the given markdown text using markdown parsing.

        Args:
            input (str): The markdown text from which to extract figures.

        Returns:
            list[FigureInfo]: A list of extracted figure information objects.
        """
        return find_figures_in_markdown(input)
