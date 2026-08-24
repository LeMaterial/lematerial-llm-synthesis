import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from llm_synthesis.utils import (
        configure_dspy,
        extract_cost_from_dspy_response,
        get_lm_cost,
    )

__all__ = [
    "configure_dspy",
    "extract_cost_from_dspy_response",
    "get_lm_cost",
]

_LAZY_SUBMODULE_BY_NAME = {
    "configure_dspy": "utils",
    "extract_cost_from_dspy_response": "utils",
    "get_lm_cost": "utils",
}


def __getattr__(name: str) -> object:
    submodule = _LAZY_SUBMODULE_BY_NAME.get(name)
    if submodule is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(f"{__name__}.{submodule}")
    return getattr(module, name)
