import csv
import json
from pathlib import Path
from typing import Any

import pandas as pd

from llm_synthesis.extraction.plots.signatures import ExtractedPlotData


def export_data_to_csv(
    extracted_data: list[ExtractedPlotData],
    output_path: str,
    include_metadata: bool = True,
) -> None:
    """
    Export extracted plot data to CSV format.

    Args:
        extracted_data: List of ExtractedPlotData objects
        output_path: Path to save the CSV file
        include_metadata: Whether to include plot metadata in the output
    """
    output_path = Path(output_path)

    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)

        # Write header
        header = ["subplot", "series_name", "x_value", "y_value", "axis"]
        if include_metadata:
            header.extend(
                ["x_label", "x_unit", "y_label", "y_unit", "plot_title"]
            )
        writer.writerow(header)

        # Write data for each subplot
        for subplot_idx, plot_data in enumerate(extracted_data):
            subplot_label = (
                plot_data.metadata.subplot_label
                or f"subplot_{subplot_idx + 1}"
            )

            for series in plot_data.data_series:
                for point in series.points:
                    row = [
                        subplot_label,
                        series.name,
                        point.x,
                        point.y,
                        point.axis,
                    ]

                    if include_metadata:
                        y_label = (
                            plot_data.metadata.left_y_axis_label
                            if point.axis == "left"
                            else plot_data.metadata.right_y_axis_label
                        )
                        y_unit = (
                            plot_data.metadata.left_y_axis_unit
                            if point.axis == "left"
                            else plot_data.metadata.right_y_axis_unit
                        )
                        row.extend(
                            [
                                plot_data.metadata.x_axis_label,
                                plot_data.metadata.x_axis_unit,
                                y_label,
                                y_unit,
                                plot_data.metadata.plot_title,
                            ]
                        )

                    writer.writerow(row)


def export_data_to_json(
    extracted_data: list[ExtractedPlotData],
    output_path: str,
    include_analysis: str | None = None,
) -> None:
    """
    Export extracted plot data to JSON format.

    Args:
        extracted_data: List of ExtractedPlotData objects
        output_path: Path to save the JSON file
        include_analysis: Optional analysis text to include
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to serializable format
    export_data = {
        "plots": [plot_data.model_dump() for plot_data in extracted_data],
        "metadata": {
            "num_subplots": len(extracted_data),
            "total_series": sum(
                len(plot.data_series) for plot in extracted_data
            ),
            "total_points": sum(
                len(series.points)
                for plot in extracted_data
                for series in plot.data_series
            ),
        },
    }

    if include_analysis:
        export_data["scientific_analysis"] = include_analysis

    with open(output_path, "w", encoding="utf-8") as jsonfile:
        json.dump(export_data, jsonfile, indent=2, default=str)


def create_dataframes(
    extracted_data: list[ExtractedPlotData],
) -> dict[str, pd.DataFrame]:
    """
    Convert extracted plot data to pandas DataFrames for easy analysis.

    Args:
        extracted_data: List of ExtractedPlotData objects

    Returns:
        Dictionary mapping subplot labels to DataFrames
    """
    dataframes = {}

    for subplot_idx, plot_data in enumerate(extracted_data):
        subplot_label = (
            plot_data.metadata.subplot_label or f"subplot_{subplot_idx + 1}"
        )

        rows = []
        for series in plot_data.data_series:
            for point in series.points:
                rows.append(
                    {
                        "series_name": series.name,
                        "x": point.x,
                        "y": point.y,
                        "axis": point.axis,
                        "color": series.color,
                        "marker_style": series.marker_style,
                        "x_label": plot_data.metadata.x_axis_label,
                        "x_unit": plot_data.metadata.x_axis_unit,
                        "y_label": (
                            plot_data.metadata.left_y_axis_label
                            if point.axis == "left"
                            else plot_data.metadata.right_y_axis_label
                        ),
                        "y_unit": (
                            plot_data.metadata.left_y_axis_unit
                            if point.axis == "left"
                            else plot_data.metadata.right_y_axis_unit
                        ),
                    }
                )

        if rows:
            dataframes[subplot_label] = pd.DataFrame(rows)

    return dataframes


def validate_extracted_data(
    extracted_data: list[ExtractedPlotData],
) -> dict[str, Any]:
    """
    Validate extracted plot data for consistency and completeness.

    Args:
        extracted_data: List of ExtractedPlotData objects

    Returns:
        Validation results dictionary
    """
    validation_results = {
        "is_valid": True,
        "warnings": [],
        "errors": [],
        "statistics": {},
    }

    total_points = 0
    total_series = 0

    for subplot_idx, plot_data in enumerate(extracted_data):
        subplot_label = (
            plot_data.metadata.subplot_label or f"subplot_{subplot_idx + 1}"
        )

        # Check metadata completeness
        if not plot_data.metadata.x_axis_label:
            validation_results["warnings"].append(
                f"{subplot_label}: Missing x-axis label"
            )

        if not plot_data.metadata.left_y_axis_label:
            validation_results["warnings"].append(
                f"{subplot_label}: Missing left y-axis label"
            )

        # Check data series
        if not plot_data.data_series:
            validation_results["errors"].append(
                f"{subplot_label}: No data series found"
            )
            validation_results["is_valid"] = False
            continue

        for series_idx, series in enumerate(plot_data.data_series):
            total_series += 1

            if not series.name:
                validation_results["warnings"].append(
                    f"{subplot_label}, series {series_idx + 1}: "
                    "Missing series name"
                )

            if not series.points:
                validation_results["errors"].append(
                    f"{subplot_label}, {series.name}: No data points found"
                )
                validation_results["is_valid"] = False
                continue

            # Check for duplicate points
            point_coords = [(p.x, p.y) for p in series.points]
            if len(point_coords) != len(set(point_coords)):
                validation_results["warnings"].append(
                    f"{subplot_label}, {series.name}: "
                    "Contains duplicate points"
                )

            # Check for reasonable coordinate ranges
            x_values = [p.x for p in series.points]
            y_values = [p.y for p in series.points]

            if any(abs(x) > 1e6 for x in x_values):
                validation_results["warnings"].append(
                    f"{subplot_label}, {series.name}: "
                    "X values seem unusually large"
                )

            if any(abs(y) > 1e6 for y in y_values):
                validation_results["warnings"].append(
                    f"{subplot_label}, {series.name}: "
                    "Y values seem unusually large"
                )

            total_points += len(series.points)

    # Statistics
    validation_results["statistics"] = {
        "total_subplots": len(extracted_data),
        "total_series": total_series,
        "total_points": total_points,
        "avg_points_per_series": total_points / max(total_series, 1),
    }

    return validation_results


def summarize_extraction_results(
    extracted_data: list[ExtractedPlotData],
    analysis: str | None = None,
) -> str:
    """
    Create a human-readable summary of extraction results.

    Args:
        extracted_data: List of ExtractedPlotData objects
        analysis: Optional scientific analysis text

    Returns:
        Formatted summary string
    """
    summary_parts = []

    # Overall statistics
    total_subplots = len(extracted_data)
    total_series = sum(len(plot.data_series) for plot in extracted_data)
    total_points = sum(
        len(series.points)
        for plot in extracted_data
        for series in plot.data_series
    )

    summary_parts.append("EXTRACTION SUMMARY")
    summary_parts.append("==================")
    summary_parts.append(f"Total subplots: {total_subplots}")
    summary_parts.append(f"Total data series: {total_series}")
    summary_parts.append(f"Total data points: {total_points}")
    summary_parts.append("")

    # Detailed breakdown
    for subplot_idx, plot_data in enumerate(extracted_data):
        subplot_label = (
            plot_data.metadata.subplot_label or f"Subplot {subplot_idx + 1}"
        )
        summary_parts.append(f"{subplot_label.upper()}:")
        summary_parts.append(
            f"  Title: {plot_data.metadata.plot_title or 'N/A'}"
        )
        summary_parts.append(
            f"  X-axis: {plot_data.metadata.x_axis_label} "
            f"({plot_data.metadata.x_axis_unit})"
        )
        summary_parts.append(
            f"  Y-axis (left): {plot_data.metadata.left_y_axis_label} "
            f"({plot_data.metadata.left_y_axis_unit})"
        )

        if plot_data.metadata.is_dual_axis:
            summary_parts.append(
                f"  Y-axis (right): {plot_data.metadata.right_y_axis_label} "
                f"({plot_data.metadata.right_y_axis_unit})"
            )

        summary_parts.append(f"  Data series: {len(plot_data.data_series)}")

        for series in plot_data.data_series:
            summary_parts.append(
                f"    - {series.name}: {len(series.points)} points "
                f"({series.axis} axis)"
            )

        if plot_data.technical_takeaways:
            summary_parts.append("  Key insights:")
            for takeaway in plot_data.technical_takeaways:
                summary_parts.append(f"    • {takeaway}")

        summary_parts.append("")

    if analysis:
        summary_parts.append("SCIENTIFIC ANALYSIS:")
        summary_parts.append("==================")
        summary_parts.append(analysis)

    return "\n".join(summary_parts)
