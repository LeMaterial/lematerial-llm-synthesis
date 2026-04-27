"""Base interface for performance linking transformers."""

from abc import abstractmethod
from typing import NamedTuple

from llm_synthesis.models.performance import SeriesMapping
from llm_synthesis.transformers.base import ExtractorInterface


class LinkingInput(NamedTuple):
    """Input for series-to-material linking.

    Attributes:
        materials: List of material names extracted from the paper
        series_names: List of series/line names from the plot
        context: Figure context (caption + surrounding text)
        plot_metadata: Dict with plot info (title, axis labels, units)
    """

    materials: list[str]
    series_names: list[str]
    context: str
    plot_metadata: dict


class PerformanceLinkingInterface(
    ExtractorInterface[LinkingInput, list[SeriesMapping]]
):
    """Interface for linking plot series to materials.

    Implementations should take plot series names and material names,
    and return mappings between them based on semantic matching.
    """

    @abstractmethod
    def forward(self, input: LinkingInput) -> list[SeriesMapping]:
        """Match plot series names to material names.

        Args:
            input: LinkingInput with materials, series names, context, and 
            metadata

        Returns:
            List of SeriesMapping objects linking series to materials
        """
        pass
