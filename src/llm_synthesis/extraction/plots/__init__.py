"""
Plot data extraction module for scientific figures.
"""

from llm_synthesis.extraction.plots.plot_parser import (
    ComprehensivePlotParser,
    PlotAnalyzer,
    PlotDataExtractor,
    PlotIdentifier,
)
from llm_synthesis.extraction.plots.signatures import (
    DataPoint,
    DataSeries,
    ExtractedPlotData,
    PlotMetadata,
)

__all__ = [
    "ComprehensivePlotParser",
    "DataPoint",
    "DataSeries",
    "ExtractedPlotData",
    "PlotAnalyzer",
    "PlotDataExtractor",
    "PlotIdentifier",
    "PlotMetadata",
]
