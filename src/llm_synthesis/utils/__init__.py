"""Utility functions for llm_synthesis.

Submodules are loaded lazily on first attribute access (PEP 562) rather
than being imported eagerly here. ``llm_synthesis.utils`` is imported
transitively any time ``llm_synthesis`` is imported (e.g. by the CLI just
to parse ``--help``), and some submodules are expensive to load —
``cost_tracking``/``dspy_utils`` pull in ``dspy`` (and, through it,
``litellm``/``transformers``), and ``figure_utils`` pulls in
``llm_synthesis.models`` — so eagerly importing all of them here made every
``llm_synthesis`` import pay for the full stack even when only a
lightweight helper like ``clean_text`` was needed.
"""

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from llm_synthesis.utils.cost_tracking import (
        extract_cost_from_dspy_response,
    )
    from llm_synthesis.utils.dspy_utils import configure_dspy, get_lm_cost
    from llm_synthesis.utils.figure_utils import (
        FigureInfo,
        clean_text_from_images,
        find_figures_in_markdown,
        insert_figure_description,
        validate_base64_image,
    )
    from llm_synthesis.utils.formula_utils import (
        extract_condition_annotation,
        find_best_material_match,
        normalize_formula,
    )
    from llm_synthesis.utils.markdown_utils import clean_text
    from llm_synthesis.utils.performance_utils import (
        aggregate_all_materials_performance,
        aggregate_performance,
        compute_linking_stats,
        get_unmatched_series,
        sanitize_filename,
    )
    from llm_synthesis.utils.prompt_utils import read_prompt_str_from_txt
    from llm_synthesis.utils.style_utils import (
        get_cmap,
        get_palette,
        set_style,
    )
    from llm_synthesis.utils.visualization import visualize_line_chart

__all__ = [
    "FigureInfo",
    "aggregate_all_materials_performance",
    "aggregate_performance",
    "clean_text",
    "clean_text_from_images",
    "compute_linking_stats",
    "configure_dspy",
    "extract_condition_annotation",
    "extract_cost_from_dspy_response",
    "find_best_material_match",
    "find_figures_in_markdown",
    "get_cmap",
    "get_lm_cost",
    "get_palette",
    "get_unmatched_series",
    "insert_figure_description",
    "normalize_formula",
    "read_prompt_str_from_txt",
    "sanitize_filename",
    "set_style",
    "validate_base64_image",
    "visualize_line_chart",
]

_LAZY_SUBMODULE_BY_NAME = {
    "extract_cost_from_dspy_response": "cost_tracking",
    "configure_dspy": "dspy_utils",
    "get_lm_cost": "dspy_utils",
    "FigureInfo": "figure_utils",
    "clean_text_from_images": "figure_utils",
    "find_figures_in_markdown": "figure_utils",
    "insert_figure_description": "figure_utils",
    "validate_base64_image": "figure_utils",
    "extract_condition_annotation": "formula_utils",
    "find_best_material_match": "formula_utils",
    "normalize_formula": "formula_utils",
    "clean_text": "markdown_utils",
    "aggregate_all_materials_performance": "performance_utils",
    "aggregate_performance": "performance_utils",
    "compute_linking_stats": "performance_utils",
    "get_unmatched_series": "performance_utils",
    "sanitize_filename": "performance_utils",
    "read_prompt_str_from_txt": "prompt_utils",
    "get_cmap": "style_utils",
    "get_palette": "style_utils",
    "set_style": "style_utils",
    "visualize_line_chart": "visualization",
}


def __getattr__(name: str) -> object:
    submodule = _LAZY_SUBMODULE_BY_NAME.get(name)
    if submodule is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(f"{__name__}.{submodule}")
    return getattr(module, name)
