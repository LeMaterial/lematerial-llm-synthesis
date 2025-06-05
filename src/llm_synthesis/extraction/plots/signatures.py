import dspy
from pydantic import BaseModel, Field


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
    right_y_axis_label: str = Field(
        description="Label of the right y-axis if dual-axis", default=""
    )
    right_y_axis_unit: str = Field(
        description="Unit of the right y-axis if dual-axis", default=""
    )
    plot_title: str = Field(description="Title of the plot", default="")
    is_dual_axis: bool = Field(description="Whether this is a dual-axis plot")
    subplot_label: str = Field(
        description="Subplot label (e.g., 'a', 'b', 'c')", default=""
    )


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


class PlotIdentificationSignature(dspy.Signature):
    """
    Signature for identifying whether a figure contains extractable scientific
    plots.
    """

    figure_base64: str = dspy.InputField(
        description="Base64 encoded image of the figure to analyze"
    )
    publication_context: str = dspy.InputField(
        description="Relevant text context from the publication "
        "about this figure"
    )

    is_extractable_plot: bool = dspy.OutputField(
        description="""Determine if this figure contains scientific plots with 
        extractable X-Y data points.
        
        CRITERIA FOR EXTRACTABLE PLOTS:
        - Contains clear X-Y coordinate systems with numerical axes
        - Shows quantitative relationships between variables
        - Has identifiable data points, lines, or curves
        - Axes have readable scales and labels
        
        INCLUDE: Line plots, scatter plots, bar charts, XRD patterns, 
        spectroscopy data, performance curves, kinetics plots, temperature 
        profiles, etc.
        
        EXCLUDE: Schematic diagrams, molecular structures, microscopy images, 
        photos, flowcharts, conceptual illustrations, journal logos, author 
        photos.
        
        Return True only if data points can be reasonably extracted."""
    )

    plot_type: str = dspy.OutputField(
        description="""Classify the type of plot if extractable. Options:
        - 'line_plot': Connected data points showing trends
        - 'scatter_plot': Individual data points without connections
        - 'bar_chart': Categorical data with bars
        - 'spectroscopy': XRD, NMR, IR, UV-Vis, etc.
        - 'kinetics': Time-dependent measurements
        - 'performance': Efficiency, conversion, selectivity curves
        - 'multiple_subplots': Multiple distinct plots in one image
        - 'other': Other extractable plot types
        - 'not_extractable': No extractable data"""
    )

    subplot_count: int = dspy.OutputField(
        description="Number of distinct subplots in the image "
        "(1 if single plot, 0 if not extractable)"
    )


class DataExtractionSignature(dspy.Signature):
    """
    Signature for extracting actual data points from identified plots.
    """

    figure_base64: str = dspy.InputField(
        description="Base64 encoded image of the extractable scientific plot"
    )
    publication_context: str = dspy.InputField(
        description="Relevant publication text providing context about the figure"
    )
    subplot_focus: str = dspy.InputField(
        description="Which subplot to focus on if multiple (e.g., 'subplot a', "
        "'all subplots', 'main plot')"
    )

    extracted_data: list[ExtractedPlotData] = dspy.OutputField(
        description="""Extract complete data from the plot(s) following these 
        guidelines:

        DATA EXTRACTION PROCESS:
        1. Identify all axis labels, units, and scales
        2. Locate all data series (different colors, markers, or line styles)
        3. Extract X-Y coordinates for each data point in each series
        4. Handle dual-axis plots by identifying which series belongs to which 
           y-axis
        5. Cross-verify extracted values against visible grid lines and axis 
           scales
        6. Provide technical insights based on the data trends
        
        ACCURACY REQUIREMENTS:
        - Read values carefully from axis scales
        - Distinguish between different data series
        - For dual-axis plots, correctly assign series to left/right axes
        - Interpolate values between grid lines when necessary
        - Include ALL visible data points, not just selected ones
        
        TECHNICAL TAKEAWAYS:
        - Identify trends (increasing, decreasing, optimal points)
        - Note correlations between variables
        - Highlight significant values or transitions
        - Explain what the data reveals about the system/process
        
        Return one ExtractedPlotData object per subplot."""
    )


class PlotAnalysisSignature(dspy.Signature):
    """
    Signature for providing detailed scientific analysis of extracted plot data
    """

    extracted_plot_data: str = dspy.InputField(
        description="JSON string of extracted plot data (ExtractedPlotData "
        "objects)"
    )
    publication_context: str = dspy.InputField(
        description="Relevant publication text providing scientific context"
    )
    figure_caption: str = dspy.InputField(
        description="Figure caption and surrounding text context"
    )

    scientific_analysis: str = dspy.OutputField(
        description="""Provide comprehensive scientific analysis of the 
        extracted data:
        
        ANALYSIS COMPONENTS:
        1. Data Quality Assessment:
           - Verify data extraction accuracy
           - Identify any potential extraction errors
           - Note data completeness and reliability
        
        2. Quantitative Analysis:
           - Calculate key metrics (slopes, maxima, minima)
           - Identify optimal operating conditions
           - Quantify performance improvements or changes
        
        3. Scientific Interpretation:
           - Explain the physical/chemical meaning of trends
           - Relate findings to the research objectives
           - Compare different data series or conditions
        
        4. Technical Insights:
           - Identify structure-property relationships
           - Note unexpected behaviors or anomalies
           - Suggest implications for practical applications
        
        FORMAT: Provide a detailed analysis (400-500 words) that would be 
        valuable for researchers interpreting this data."""
    )
