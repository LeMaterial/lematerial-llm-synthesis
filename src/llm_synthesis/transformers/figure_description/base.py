from llm_synthesis.models.figure import FigureInfoWithPaper
from llm_synthesis.transformers.base import ExtractorInterface

FigureDescriptionExtractorInterface = ExtractorInterface[
    FigureInfoWithPaper, str
]
