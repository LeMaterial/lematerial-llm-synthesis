from llm_synthesis.models.figure import FigureInfo
from llm_synthesis.models.paper import ImageData
from llm_synthesis.transformers.figure_extraction.base import (
    FigureFilterInterface,
    FilteredImageFigureExtractorInterface,
    ImageFigureExtractorInterface,
)


class FilteredImageFigureExtractor(FilteredImageFigureExtractorInterface):
    """
    Extracts figures from image bytes and filters them in one step.
    """

    def __init__(
        self,
        image_figure_extractor: ImageFigureExtractorInterface,
        figure_filter: FigureFilterInterface,
    ):
        """
        Initialize the filtered image figure extractor.

        Args:
            image_figure_extractor: The extractor to use for
                extracting figures from image bytes
            figure_filter: The filter to apply to the extracted figures
        """
        self.image_figure_extractor = image_figure_extractor
        self.figure_filter = figure_filter

    def forward(self, input: list[ImageData]) -> list[FigureInfo]:
        """
        Extract figures from image bytes and filter them.

        Args:
            input: The image data to extract figures from

        Returns:
            List of filtered figures
        """
        # Extract all figures
        all_figures = self.image_figure_extractor.forward(input)

        # Filter the figures
        filtered_figures = self.figure_filter.forward(all_figures)

        return filtered_figures
