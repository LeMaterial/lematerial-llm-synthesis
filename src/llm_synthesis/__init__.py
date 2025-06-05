# TODO: add public API here -- fucntions to be accessible to public

# Public API exports
from llm_synthesis.extraction.plots import (
    ComprehensivePlotParser,
    ExtractedPlotData,
)
from llm_synthesis.utils import (
    EnhancedMarkdownProcessor,
    EnhancedPlotProcessor,
    configure_dspy,
    extract_markdown,
    process_paper_with_figure_descriptions,
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
