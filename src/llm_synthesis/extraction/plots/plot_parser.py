import json
from typing import Any

import dspy

from llm_synthesis.extraction.plots.signatures import (
    DataExtractionSignature,
    ExtractedPlotData,
    PlotAnalysisSignature,
    PlotIdentificationSignature,
)
from llm_synthesis.utils.figure_utils import clean_text_from_images


class PlotIdentifier(dspy.Module):
    """
    Module for identifying whether figures contain extractable scientific plots
    """

    def __init__(self):
        self.predict = dspy.Predict(PlotIdentificationSignature)

    def __call__(
        self,
        figure_base64: str,
        publication_context: str = "",
    ) -> dict[str, Any]:
        """
        Identify if a figure contains extractable plots.

        Args:
            figure_base64: Base64 encoded figure image
            publication_context: Relevant text context about the figure

        Returns:
            Dict containing identification results
        """
        # Clean context text to reduce token usage
        clean_context = clean_text_from_images(publication_context)

        result = self.predict(
            figure_base64=figure_base64,
            publication_context=clean_context,
        )

        return {
            "is_extractable": result.is_extractable_plot,
            "plot_type": result.plot_type,
            "subplot_count": result.subplot_count,
        }


class PlotDataExtractor(dspy.Module):
    """
    Module for extracting data points from identified plots.
    """

    def __init__(self):
        self.predict = dspy.Predict(DataExtractionSignature)

    def __call__(
        self,
        figure_base64: str,
        publication_context: str = "",
        subplot_focus: str = "all subplots",
    ) -> list[ExtractedPlotData]:
        """
        Extract data points from scientific plots.

        Args:
            figure_base64: Base64 encoded figure image
            publication_context: Relevant text context about the figure
            subplot_focus: Which subplot(s) to focus on

        Returns:
            List of ExtractedPlotData objects
        """
        # Clean context text to reduce token usage
        clean_context = clean_text_from_images(publication_context)

        result = self.predict(
            figure_base64=figure_base64,
            publication_context=clean_context,
            subplot_focus=subplot_focus,
        )

        return result.extracted_data


class PlotAnalyzer(dspy.Module):
    """
    Module for providing detailed scientific analysis of extracted plot data.
    """

    def __init__(self):
        self.predict = dspy.Predict(PlotAnalysisSignature)

    def __call__(
        self,
        extracted_plot_data: list[ExtractedPlotData],
        publication_context: str = "",
        figure_caption: str = "",
    ) -> str:
        """
        Analyze extracted plot data and provide scientific insights.

        Args:
            extracted_plot_data: List of extracted plot data
            publication_context: Relevant publication text
            figure_caption: Figure caption and context

        Returns:
            Detailed scientific analysis string
        """
        # Convert extracted data to JSON string for processing
        data_json = json.dumps(
            [data.model_dump() for data in extracted_plot_data],
            indent=2,
            default=str,
        )

        # Clean context text
        clean_context = clean_text_from_images(publication_context)
        clean_caption = clean_text_from_images(figure_caption)

        result = self.predict(
            extracted_plot_data=data_json,
            publication_context=clean_context,
            figure_caption=clean_caption,
        )

        return result.scientific_analysis.strip()


class ComprehensivePlotParser(dspy.Module):
    """
    Comprehensive plot parser that combines identification, extraction, and 
    analysis.
    """

    def __init__(self):
        self.identifier = PlotIdentifier()
        self.extractor = PlotDataExtractor()
        self.analyzer = PlotAnalyzer()

    def process_figure(
        self,
        figure_base64: str,
        publication_context: str = "",
        figure_caption: str = "",
        extract_data: bool = True,
        analyze_data: bool = True,
    ) -> dict[str, Any]:
        """
        Complete processing pipeline for a single figure.

        Args:
            figure_base64: Base64 encoded figure image
            publication_context: Relevant publication text
            figure_caption: Figure caption and context
            extract_data: Whether to extract data points
            analyze_data: Whether to perform scientific analysis

        Returns:
            Complete processing results
        """
        results = {}

        # Step 1: Identify if figure contains extractable plots
        print("Identifying plot type...")
        identification = self.identifier(
            figure_base64=figure_base64,
            publication_context=publication_context,
        )
        results["identification"] = identification

        if not identification["is_extractable"]:
            print("Figure does not contain extractable plots.")
            return results

        print(
            f"Identified {identification['plot_type']} with "
            f"{identification['subplot_count']} subplot(s)"
        )

        if not extract_data:
            return results

        # Step 2: Extract data points
        print("Extracting data points...")
        try:
            extracted_data = self.extractor(
                figure_base64=figure_base64,
                publication_context=publication_context,
                subplot_focus="all subplots",
            )
            results["extracted_data"] = extracted_data

            # Print summary of extracted data
            for i, plot_data in enumerate(extracted_data):
                series_count = len(plot_data.data_series)
                total_points = sum(
                    len(series.points) for series in plot_data.data_series
                )
                print(
                    f"  Subplot {i + 1}: {series_count} data series, "
                    f"{total_points} total points"
                )

        except Exception as e:
            print(f"Error during data extraction: {e!s}")
            results["extraction_error"] = str(e)
            return results

        if not analyze_data or "extracted_data" not in results:
            return results

        # Step 3: Analyze extracted data
        print("Generating scientific analysis...")
        try:
            analysis = self.analyzer(
                extracted_plot_data=extracted_data,
                publication_context=publication_context,
                figure_caption=figure_caption,
            )
            results["analysis"] = analysis

        except Exception as e:
            print(f"Error during analysis: {e!s}")
            results["analysis_error"] = str(e)

        return results

    def __call__(
        self,
        figure_base64: str,
        publication_context: str = "",
        figure_caption: str = "",
    ) -> dict[str, Any]:
        """Convenience method for complete processing."""
        return self.process_figure(
            figure_base64=figure_base64,
            publication_context=publication_context,
            figure_caption=figure_caption,
        )
