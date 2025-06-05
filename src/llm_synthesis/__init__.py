# TODO: add public API here -- functions to be accessible to public

# Public API exports
from llm_synthesis.extraction.plots.plot_parser import ComprehensivePlotParser
from llm_synthesis.extraction.plots.signatures import ExtractedPlotData
from llm_synthesis.utils import (
    EnhancedMarkdownProcessor,
    configure_dspy,
    extract_markdown,
    process_paper_with_figure_descriptions,
)
from llm_synthesis.utils.enhanced_plot_processor import (
    EnhancedPlotProcessor,
    process_paper_with_plot_extraction,
)

__all__ = [
    "ComprehensivePlotParser",
    "EnhancedMarkdownProcessor",
    "EnhancedPlotProcessor",
    "ExtractedPlotData",
    "configure_dspy",
    "extract_markdown",
    "process_paper_with_figure_descriptions",
    "process_paper_with_plot_extraction",
]
