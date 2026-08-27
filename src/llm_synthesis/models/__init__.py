"""Data models for llm_synthesis.

``FigureSegmenter`` (dino), ``FlorenceSegmenter``/``Detection`` (florence),
and ``FigureClassifier`` (resnet) pull in torch/transformers/sklearn. They
are loaded lazily on first attribute access (PEP 562) so that importing
``llm_synthesis.models`` — which happens transitively any time
``llm_synthesis`` is imported — stays cheap for callers that only need
lightweight models like ``Paper`` or the performance-linking types below.
"""

import importlib
from typing import TYPE_CHECKING

from llm_synthesis.models.performance import (
    LinkingStats,
    MaterialPerformanceData,
    MaterialPlotEntry,
    PlotMaterialMapping,
    SeriesMapping,
)

if TYPE_CHECKING:
    from llm_synthesis.models.dino import FigureSegmenter
    from llm_synthesis.models.florence import Detection, FlorenceSegmenter
    from llm_synthesis.models.resnet import FigureClassifier

__all__ = [
    "Detection",
    "FigureClassifier",
    "FigureSegmenter",
    "FlorenceSegmenter",
    "LinkingStats",
    "MaterialPerformanceData",
    "MaterialPlotEntry",
    "PlotMaterialMapping",
    "SeriesMapping",
]

_LAZY_SUBMODULE_BY_NAME = {
    "FigureSegmenter": "dino",
    "Detection": "florence",
    "FlorenceSegmenter": "florence",
    "FigureClassifier": "resnet",
}


def __getattr__(name: str) -> object:
    submodule = _LAZY_SUBMODULE_BY_NAME.get(name)
    if submodule is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(f"{__name__}.{submodule}")
    return getattr(module, name)
