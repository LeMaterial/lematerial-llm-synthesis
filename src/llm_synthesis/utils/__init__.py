from llm_synthesis.utils.dspy_utils import configure_dspy
from llm_synthesis.utils.enhanced_markdown_processor import (
    EnhancedMarkdownProcessor,
    process_paper_with_figure_descriptions,
)
from llm_synthesis.utils.enhanced_plot_processor import (
    EnhancedPlotProcessor,
    process_paper_with_plot_extraction,
)
from llm_synthesis.utils.figure_utils import (
    FigureInfo,
    clean_text_from_images,
    find_figures_in_markdown,
    insert_figure_description,
    validate_base64_image,
)
from llm_synthesis.utils.markdown_utils import remove_figs
from llm_synthesis.utils.parse_utils import (
    change_extension,
    docling_markdown,
    extract_markdown,
    mistral_markdown,
)
from llm_synthesis.utils.plot_utils import (
    create_dataframes,
    export_data_to_csv,
    export_data_to_json,
    summarize_extraction_results,
    validate_extracted_data,
)

# Add to existing __all__ list:
__all__ = [
    "change_extension",
    "clean_text_from_images",
    "configure_dspy",
    "create_dataframes",
    "docling_markdown",
    "EnhancedMarkdownProcessor",
    "EnhancedPlotProcessor",
    "export_data_to_csv",
    "export_data_to_json",
    "extract_markdown",
    "FigureInfo",
    "find_figures_in_markdown",
    "insert_figure_description",
    "mistral_markdown",
    "process_paper_with_figure_descriptions",
    "process_paper_with_plot_extraction",
    "remove_figs",
    "summarize_extraction_results",
    "validate_base64_image",
    "validate_extracted_data",
]