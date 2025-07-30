from llm_synthesis.transformers.figure_extraction import regex_figure_extractor
from llm_synthesis.transformers.figure_extraction.figure_filter import (
    FigureFilter,
    PlotFigureFilter,
    QuantitativeFigureFilter,
)
from llm_synthesis.transformers.figure_extraction.filtered_figure_extractor import (  # noqa: E501
    FilteredFigureExtractor,
)
from llm_synthesis.transformers.figure_extraction.filtered_image_figure_extractor import (  # noqa: E501
    FilteredImageFigureExtractor,
)
from llm_synthesis.transformers.figure_extraction.image_figure_extractor import (  # noqa: E501
    ImageFigureExtractor,
)

FigureExtractorMarkdown = regex_figure_extractor.FigureExtractorMarkdown

__all__ = [
    "FigureExtractorMarkdown",
    "FigureFilter",
    "FilteredFigureExtractor",
    "FilteredImageFigureExtractor",
    "ImageFigureExtractor",
    "PlotFigureFilter",
    "QuantitativeFigureFilter",
]
