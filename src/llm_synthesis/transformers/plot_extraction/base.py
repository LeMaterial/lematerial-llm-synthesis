from llm_synthesis.models.figure import FigureInfoWithPaper
from llm_synthesis.models.plot import (
    ExtractedLinePlotData,
    ExtractedPlotData,
    PlotInfo,
)
from llm_synthesis.transformers.base import ExtractorInterface

PlotInformationExtractorInterface = ExtractorInterface[
    FigureInfoWithPaper, PlotInfo
]

PlotDataExtractorInterface = ExtractorInterface[
    tuple[FigureInfoWithPaper, str], list[ExtractedPlotData]
]

PlotAnalysisSignature = ExtractorInterface[
    tuple[FigureInfoWithPaper, ExtractedPlotData], str
]

LinePlotDataExtractorInterface = ExtractorInterface[
    FigureInfoWithPaper, ExtractedLinePlotData
]
