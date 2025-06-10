from llm_synthesis.models.figure import FigureInfo
from llm_synthesis.transformers.base import ExtractorInterface

FigureExtractorInterface = ExtractorInterface[str, list[FigureInfo]]
