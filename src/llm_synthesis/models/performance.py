"""Performance linking models for synthesis-performance data integration."""

from pydantic import BaseModel, Field


class SeriesMapping(BaseModel):
    """A single mapping from a plot series name to a material name."""

    series_name: str = Field(description="Series/line name from the plot legend")
    material_name: str = Field(
        description="Matched material name from synthesis extraction"
    )
    confidence: str = Field(
        default="medium", description="Confidence level: 'high', 'medium', or 'low'"
    )
    reasoning: str = Field(default="", description="Explanation for this match")


class PlotMaterialMapping(BaseModel):
    """All series-to-material mappings for a single plot."""

    plot_index: int = Field(description="Index of the plot in the paper's plot list")
    figure_reference: str = Field(default="", description="e.g. 'Fig. 3a'")
    mappings: list[SeriesMapping] = Field(default_factory=list)
    unmatched_series: list[str] = Field(
        default_factory=list,
        description="Series that could not be matched (baselines, references, etc.)",
    )


class MaterialPlotEntry(BaseModel):
    """One plot series linked to a material, with its coordinate data."""

    plot_index: int
    figure_reference: str = ""
    series_name: str
    coordinates: list[list[float]] = Field(default_factory=list)
    x_axis_label: str | None = None
    x_axis_unit: str | None = None
    y_axis_label: str | None = None
    y_axis_unit: str | None = None
    plot_title: str | None = None
    confidence: str = "medium"


class MaterialPerformanceData(BaseModel):
    """All performance data for a single material, aggregated across plots."""

    material_name: str
    plot_data: list[MaterialPlotEntry] = Field(default_factory=list)


class LinkingStats(BaseModel):
    """Statistics about plot linking for summary output."""

    total_plots_extracted: int = 0
    plots_linked: int = 0
    plots_skipped_not_relevant_x: int = 0
    plots_skipped_not_relevant_y: int = 0
    plots_skipped_no_series: int = 0
    skipped_plots: list[dict] = Field(default_factory=list)
    linked_plots: list[dict] = Field(default_factory=list)
    all_unmatched_series: list[str] = Field(default_factory=list)
    confidence_counts: dict[str, int] = Field(
        default_factory=lambda: {"high": 0, "medium": 0, "low": 0}
    )
