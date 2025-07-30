from llm_synthesis.models.figure import FigureInfo
from llm_synthesis.models.paper import ImageData
from llm_synthesis.transformers.base import ExtractorInterface

FigureExtractorInterface = ExtractorInterface[str, list[FigureInfo]]

# New interfaces for the filtering pipeline
FigureFilterInterface = ExtractorInterface[list[FigureInfo], list[FigureInfo]]

# Combined interface that goes from text directly to filtered figures
FilteredFigureExtractorInterface = ExtractorInterface[str, list[FigureInfo]]

# Interface for extracting figures directly from image bytes (for HF datasets)
ImageFigureExtractorInterface = ExtractorInterface[
    list[ImageData], list[FigureInfo]
]

# Combined interface that goes from image bytes directly to filtered figures
FilteredImageFigureExtractorInterface = ExtractorInterface[
    list[ImageData], list[FigureInfo]
]
