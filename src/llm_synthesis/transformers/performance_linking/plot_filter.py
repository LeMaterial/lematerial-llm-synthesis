"""Plot filter for which plots are relevant for performance linking."""

import logging
from typing import NamedTuple

from llm_synthesis.config.plot_filter_config import PlotFilterConfig
from llm_synthesis.models.plot import ExtractedLinePlotData

logger = logging.getLogger(__name__)


class FilterResult(NamedTuple):
    """Result of plot filtering.

    Attributes:
        is_relevant: Whether the plot should be included
        skip_reason: If not relevant, the reason for skipping (None if relevant)
    """

    is_relevant: bool
    skip_reason: str | None


class PlotFilter:
    """Filter for determining which plots are relevant for performance linking.

    Uses configurable criteria to determine if a plot contains performance
    data worth linking to materials. This helps avoid linking characterization
    plots (XRD, TGA, SEM, etc.) that don't represent material performance.

    Attributes:
        config: PlotFilterConfig with filtering criteria
    """

    def __init__(self, config: PlotFilterConfig | None = None):
        """Initialize the filter.

        Args:
            config: PlotFilterConfig. If None, uses default catalysis config.
        """
        self.config = config or PlotFilterConfig()

    def filter(self, plot: ExtractedLinePlotData) -> FilterResult:
        """Determine if a plot should be included in performance linking.

        Args:
            plot: Extracted plot data with axis labels, units, and series

        Returns:
            FilterResult with is_relevant flag and optional skip_reason
        """
        # Check if plot has any series
        if not plot.name_to_coordinates:
            return FilterResult(
                is_relevant=False,
                skip_reason="no_series",
            )

        # Check x-axis
        if self.config.filter_x_axis:
            if not self.config.is_relevant_x_axis(
                plot.x_axis_label, plot.x_axis_unit
            ):
                return FilterResult(
                    is_relevant=False,
                    skip_reason="not_relevant_x",
                )

        # Check y-axis
        if self.config.filter_y_axis:
            if not self.config.is_relevant_y_axis(
                plot.y_left_axis_label, plot.y_left_axis_unit
            ):
                return FilterResult(
                    is_relevant=False,
                    skip_reason="not_relevant_y",
                )

        return FilterResult(is_relevant=True, skip_reason=None)

    def filter_plots(
        self,
        plots: list[ExtractedLinePlotData],
        log_skipped: bool = True,
    ) -> tuple[list[tuple[int, ExtractedLinePlotData]], dict[str, int]]:
        """Filter a list of plots and return relevant ones with stats.

        Args:
            plots: List of extracted plot data
            log_skipped: Whether to log info about skipped plots

        Returns:
            Tuple of:
                - List of (index, plot) tuples for relevant plots
                - Dict of skip reason counts
        """
        relevant_plots = []
        skip_counts = {
            "no_series": 0,
            "not_relevant_x": 0,
            "not_relevant_y": 0,
        }

        for idx, plot in enumerate(plots):
            result = self.filter(plot)

            if result.is_relevant:
                relevant_plots.append((idx, plot))
            else:
                if result.skip_reason:
                    skip_counts[result.skip_reason] = (
                        skip_counts.get(result.skip_reason, 0) + 1
                    )

                if log_skipped:
                    logger.info(
                        f"  Skipping plot {idx} '{plot.title or 'N/A'}' "
                        f"(reason: {result.skip_reason}, "
                        f"x-axis: '{plot.x_axis_label}' [{plot.x_axis_unit}], "
                        f"y-axis: '{plot.y_left_axis_label}' "
                        f"[{plot.y_left_axis_unit}])"
                    )

        return relevant_plots, skip_counts
