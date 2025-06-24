from pydantic import BaseModel, Field


class PlotInfo(BaseModel):
    """Information about a plot found in markdown text."""

    plot_type: str
    subplot_count: int
    is_extractable_plot: bool


class DataPoint(BaseModel):
    """Represents a single data point with x and y coordinates."""

    x: float = Field(description="X-coordinate value")
    y: float = Field(description="Y-coordinate value")
    series_name: str = Field(
        description="Name of the data series this point belongs to"
    )
    axis: str = Field(
        description="Which y-axis this point belongs to "
        "(left/right for dual-axis plots)"
    )


class DataSeries(BaseModel):
    """Represents a complete data series."""

    name: str = Field(description="Name/label of the data series")
    points: list[DataPoint] = Field(
        description="List of data points in this series"
    )
    axis: str = Field(
        description="Which y-axis this series belongs to (left/right)"
    )
    color: str = Field(
        description="Color of the data series if identifiable",
        default="unknown",
    )
    marker_style: str = Field(
        description="Marker style (circle, square, triangle, etc.)",
        default="unknown",
    )


class PlotMetadata(BaseModel):
    """Metadata about the plot structure."""

    x_axis_label: str = Field(description="Label of the x-axis")
    x_axis_unit: str = Field(description="Unit of the x-axis", default="")
    left_y_axis_label: str = Field(description="Label of the left y-axis")
    left_y_axis_unit: str = Field(
        description="Unit of the left y-axis", default=""
    )
    right_y_axis_label: str | None = Field(
        description="Label of the right y-axis if dual-axis", default=""
    )
    right_y_axis_unit: str | None = Field(
        description="Unit of the right y-axis if dual-axis", default=""
    )
    plot_title: str = Field(description="Title of the plot", default="")
    is_dual_axis: bool = Field(description="Whether this is a dual-axis plot")
    # We will probably chunk plots into subplots and parse them separately?
    # subplot_label: str = Field(
    #     description="Subplot label (e.g., 'a', 'b', 'c')", default=""
    # )


class ExtractedPlotData(BaseModel):
    """Complete extracted data from a plot."""

    metadata: PlotMetadata = Field(
        description="Plot metadata including axis labels and titles"
    )
    data_series: list[DataSeries] = Field(
        description="All data series found in the plot"
    )
    technical_takeaways: list[str] = Field(
        description="Key technical/scientific insights from the plot"
    )


class ClaudeExtractedPlotData(BaseModel):
    name_to_coordinates: dict[str, list[list[float]]]
    title: str | None
    x_axis_label: str | None
    x_axis_unit: str | None
    y_left_axis_label: str | None
    y_left_axis_unit: str | None
