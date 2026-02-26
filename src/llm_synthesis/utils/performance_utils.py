"""Utility functions for performance data aggregation and processing."""

import logging

from llm_synthesis.models.performance import (
    LinkingStats,
    MaterialPerformanceData,
    MaterialPlotEntry,
    PlotMaterialMapping,
)
from llm_synthesis.models.plot import ExtractedLinePlotData

logger = logging.getLogger(__name__)


def aggregate_performance(
    material_name: str,
    mappings: list[PlotMaterialMapping],
    plots: list[ExtractedLinePlotData],
) -> MaterialPerformanceData:
    """Collect all performance data for a single material across all plots.

    Args:
        material_name: Name of the material to aggregate data for
        mappings: List of PlotMaterialMapping from linking
        plots: List of ExtractedLinePlotData (indexed by plot_index)

    Returns:
        MaterialPerformanceData with all performance entries for the material
    """
    entries = []

    for mapping in mappings:
        plot = plots[mapping.plot_index]
        for sm in mapping.mappings:
            if sm.material_name == material_name:
                coords = plot.name_to_coordinates.get(sm.series_name, [])
                entries.append(
                    MaterialPlotEntry(
                        plot_index=mapping.plot_index,
                        figure_reference=mapping.figure_reference,
                        series_name=sm.series_name,
                        coordinates=coords,
                        x_axis_label=plot.x_axis_label,
                        x_axis_unit=plot.x_axis_unit,
                        y_axis_label=plot.y_left_axis_label,
                        y_axis_unit=plot.y_left_axis_unit,
                        plot_title=plot.title,
                        confidence=sm.confidence,
                    )
                )

    return MaterialPerformanceData(
        material_name=material_name,
        plot_data=entries,
    )


def aggregate_all_materials_performance(
    materials: list[str],
    mappings: list[PlotMaterialMapping],
    plots: list[ExtractedLinePlotData],
) -> dict[str, MaterialPerformanceData]:
    """Aggregate performance data for all materials.

    Args:
        materials: List of material names
        mappings: List of PlotMaterialMapping from linking
        plots: List of ExtractedLinePlotData

    Returns:
        Dict mapping material name to MaterialPerformanceData
    """
    performance_data = {}
    for mat in materials:
        perf = aggregate_performance(mat, mappings, plots)
        if perf.plot_data:  # Only include materials with performance data
            performance_data[mat] = perf
    return performance_data


def compute_linking_stats(
    total_plots: int,
    mappings: list[PlotMaterialMapping],
    skip_counts: dict[str, int],
    skipped_plots: list[dict],
) -> LinkingStats:
    """Compute statistics from linking results.

    Args:
        total_plots: Total number of plots extracted
        mappings: List of PlotMaterialMapping from linking
        skip_counts: Dict of skip reason to count
        skipped_plots: List of dicts with skipped plot details

    Returns:
        LinkingStats with comprehensive statistics
    """
    stats = LinkingStats(
        total_plots_extracted=total_plots,
        plots_linked=len(mappings),
        plots_skipped_not_relevant_x=skip_counts.get("not_relevant_x", 0),
        plots_skipped_not_relevant_y=skip_counts.get("not_relevant_y", 0),
        plots_skipped_no_series=skip_counts.get("no_series", 0),
        skipped_plots=skipped_plots,
    )

    # Aggregate stats from mappings
    for mapping in mappings:
        # Track unmatched series
        stats.all_unmatched_series.extend(mapping.unmatched_series)

        # Count confidence levels
        for m in mapping.mappings:
            conf = m.confidence.lower() if m.confidence else "medium"
            if conf in stats.confidence_counts:
                stats.confidence_counts[conf] += 1

        # Add linked plot details
        stats.linked_plots.append(
            {
                "plot_index": mapping.plot_index,
                "figure_reference": mapping.figure_reference,
                "series_count": len(mapping.mappings)
                + len(mapping.unmatched_series),
                "matched_count": len(mapping.mappings),
                "unmatched_count": len(mapping.unmatched_series),
            }
        )

    return stats


def get_unmatched_series(mappings: list[PlotMaterialMapping]) -> list[str]:
    """Get all unmatched series names across all plots.

    Args:
        mappings: List of PlotMaterialMapping

    Returns:
        List of series names that were not matched to any material
    """
    return [s for m in mappings for s in m.unmatched_series]


def sanitize_filename(name: str) -> str:
    """Turn a material name into a safe filename.

    Args:
        name: Material name (e.g., "10%Ni/Al2O3")

    Returns:
        Safe filename string (e.g., "10pctNi_Al2O3")
    """
    return name.replace("/", "_").replace(" ", "_").replace("%", "pct")
