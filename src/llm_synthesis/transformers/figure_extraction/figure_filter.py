from collections.abc import Callable

from llm_synthesis.models.figure import FigureInfo
from llm_synthesis.transformers.figure_extraction.base import (
    FigureFilterInterface,
)


class FigureFilter(FigureFilterInterface):
    """
    A filter that can filter figures based on various criteria.
    """

    def __init__(
        self,
        filter_function: Callable[[FigureInfo], bool] | None = None,
        quantitative_only: bool = True,
        allowed_figure_classes: list[str] | None = None,
        min_position: int | None = None,
        max_position: int | None = None,
    ):
        """
        Initialize the figure filter.

        Args:
            filter_function: filter fn that takes a FigureInfo and returns bool
            quantitative_only: If True, only keep figures marked as quantitative
            allowed_figure_classes: List of figure classes to keep
                (e.g., ["Bar plots", "Graph plots"])
            min_position: Minimum position in text to keep
            max_position: Maximum position in text to keep
        """
        self.filter_function = filter_function
        self.quantitative_only = quantitative_only
        self.allowed_figure_classes = allowed_figure_classes or []
        self.min_position = min_position
        self.max_position = max_position

    def forward(self, input: list[FigureInfo]) -> list[FigureInfo]:
        """
        Filter the input figures based on the configured criteria.

        Args:
            input: List of figures to filter

        Returns:
            List of filtered figures
        """
        filtered_figures = []

        for figure in input:
            if self._should_keep_figure(figure):
                filtered_figures.append(figure)

        return filtered_figures

    def _should_keep_figure(self, figure: FigureInfo) -> bool:
        """
        Determine if a figure should be kept based on the filter criteria.

        Args:
            figure: The figure to evaluate

        Returns:
            True if the figure should be kept, False otherwise
        """
        # Apply custom filter function if provided
        if self.filter_function is not None:
            if not self.filter_function(figure):
                return False

        # Filter by quantitative status
        if self.quantitative_only and not figure.quantitative:
            return False

        # Filter by figure class
        if (
            self.allowed_figure_classes
            and figure.figure_class not in self.allowed_figure_classes
        ):
            return False

        # Filter by position
        if (
            self.min_position is not None
            and figure.position < self.min_position
        ):
            return False

        if (
            self.max_position is not None
            and figure.position > self.max_position
        ):
            return False

        return True


class QuantitativeFigureFilter(FigureFilter):
    """
    A specialized filter that only keeps quantitative figures.
    """

    def __init__(self):
        super().__init__(quantitative_only=True)


class PlotFigureFilter(FigureFilter):
    """
    A specialized filter that only keeps figures that are likely to be plots.
    """

    def __init__(self):
        plot_classes = [
            "Bar plots",
            # "Contour plot",
            # "Graph plots",
            "Scatter plot",
            "Line plots",
            # "Tables",
        ]
        super().__init__(
            quantitative_only=True, allowed_figure_classes=plot_classes
        )
