"""Abstract base classes for domain-specific metric extraction.

Domains declare two optional extraction passes that run after the standard
SynthesisPerformancePipeline:

  BaseTextMetricExtractor
    An extra LLM call on the paper text to extract domain-specific scalars
    (e.g. Tc values from superconductivity papers).  Returns a per-material
    dict that is passed to the output writer alongside the pipeline result.

  BaseVLMMetricProcessor
    An extra VLM pass on the already-filtered relevant plots to extract
    domain-specific quantities (e.g. geometric Tc construction from R(T)
    plots).  Receives the full linking context so it can cross-reference
    synthesis and performance data.

Both are optional.  Set them to None in DomainConfig to skip.
"""

from abc import ABC, abstractmethod
from typing import Any

from llm_synthesis.models.figure import FigureInfo
from llm_synthesis.models.performance import PlotMaterialMapping
from llm_synthesis.models.plot import ExtractedLinePlotData


class BaseTextMetricExtractor(ABC):
    """Extract domain-specific scalar metrics from paper text.

    Called once per paper, before VLM post-processing.
    """

    @abstractmethod
    def extract(
        self,
        paper_text: str,
        materials: list[str],
    ) -> dict[str, Any]:
        """Run extraction and return per-material metrics.

        Args:
            paper_text: Full cleaned paper text.
            materials: List of material names extracted by the pipeline.

        Returns:
            Dict mapping material name -> domain metric dict.
            Materials not present in the output are treated as having no data.
            Example for superconductors::

                {
                    "Cu0.5Ba0.5": {
                        "superconducting": True,
                        "T_onset": 92.1,
                        "Tc_mid": 90.3,
                        "T_zero": 88.5,
                    },
                    ...
                }
        """


class BaseVLMMetricProcessor(ABC):
    """Extract domain-specific metrics from relevant plots via a VLM.

    Called once per paper after the pipeline has filtered plots and linked
    series to materials.  Receives the full linking context so implementations
    can cross-reference synthesis data if needed.
    """

    @abstractmethod
    def process(
        self,
        relevant_plots: list[tuple[int, ExtractedLinePlotData]],
        plot_figures: list[FigureInfo],
        plot_mappings: list[PlotMaterialMapping],
        materials: list[str],
        paper_text: str,
    ) -> dict[str, Any]:
        """Run VLM processing and return per-material metrics.

        Args:
            relevant_plots: List of (original_index, plot_data) tuples for
                plots that passed the domain's PlotFilterConfig.
            plot_figures: FigureInfo objects corresponding to *all* extracted
                plots (indexed by original plot index, same as pipeline).
            plot_mappings: Series-to-material mappings produced by the linker.
            materials: List of material names.
            paper_text: Full paper text (for context).

        Returns:
            Dict mapping material name -> domain metric dict.
            Example for superconductors::

                {
                    "Cu0.5Ba0.5": {
                        "superconducting": True,
                        "T_onset": 92.1,
                        "Tc_mid": 90.3,
                        "T_zero": 88.5,
                        "Delta_Tc": 3.6,
                        "source": "main plot",
                    },
                    ...
                }
        """
