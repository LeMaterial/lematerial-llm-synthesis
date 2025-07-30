from llm_synthesis.models.figure import FigureInfo
from llm_synthesis.transformers.figure_extraction.base import (
    FigureExtractorInterface,
    FigureFilterInterface,
    FilteredFigureExtractorInterface,
)


class FilteredFigureExtractor(FilteredFigureExtractorInterface):
    """
    Extracts and filters figures in one step.
    """

    def __init__(
        self,
        figure_extractor: FigureExtractorInterface,
        figure_filter: FigureFilterInterface,
    ):
        """
        Initialize the filtered figure extractor.

        Args:
            figure_extractor: extractor to use for extracting figures
            figure_filter: filter to apply to the extracted figures
        """
        self.figure_extractor = figure_extractor
        self.figure_filter = figure_filter

    def forward(self, input: str) -> list[FigureInfo]:
        """
        Extract figures from text and filter them.

        Args:
            input: text to extract figures from

        Returns:
            List of filtered figures
        """
        # Extract all figures
        all_figures = self.figure_extractor.forward(input)

        # Filter the figures
        filtered_figures = self.figure_filter.forward(all_figures)

        return filtered_figures
