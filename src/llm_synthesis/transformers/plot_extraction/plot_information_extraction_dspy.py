import dspy

from llm_synthesis.models.figure import FigureInfoWithPaper
from llm_synthesis.models.plot import PlotInfo
from llm_synthesis.transformers.plot_extraction.base import (
    PlotInformationExtractorInterface,
)
from llm_synthesis.utils.figure_utils import clean_text_from_images


class PlotInformationExtractor(PlotInformationExtractorInterface):
    """
    Extractor that uses dspy to extract plot information from figures.
    """

    def __init__(self, signature: type[dspy.Signature], lm: dspy.LM):
        """
        Initialize the extractor with a dspy signature and language model.

        Args:
            signature (dspy.Signature): The dspy signature specifying
                input/output fields.
            lm (dspy.LM): The language model to use for prediction.
        """
        self._validate_signature(signature)
        self.signature = signature
        self.lm = lm

    def forward(self, input: FigureInfoWithPaper) -> PlotInfo:
        """
        Extract plot information using the language model and signature.

        Args:
            input (FigureInfoWithPaper): The figure and paper context.

        Returns:
            PlotInfo: The extracted plot information.
        """
        predict_kwargs = {
            "figure_base64": input.base64_data,
            "publication_context": clean_text_from_images(input.paper_text),
        }
        with dspy.settings.context(lm=self.lm):
            result = dspy.Predict(self.signature)(**predict_kwargs)
            return PlotInfo(
                plot_type=result.plot_type,
                subplot_count=result.subplot_count,
                is_extractable_plot=result.is_extractable_plot,
            )

    def _validate_signature(self, signature: type[dspy.Signature]):
        """
        Validate that the signature contains all required input and output
        fields with correct types.

        Args:
            signature (dspy.Signature): The signature to validate.

        Raises:
            ValueError: If any required field is missing or has the wrong type.
        """
        required_input_fields = [
            "figure_base64",
            "publication_context",
        ]

        for field in required_input_fields:
            if field not in signature.input_fields:
                raise ValueError(f"{field} must be in signature")
            if signature.input_fields[field].annotation is not str:
                raise ValueError(f"{field} must be a string")

        required_output_fields = [
            "plot_type",
            "subplot_count",
            "is_extractable_plot",
        ]
        for field in required_output_fields:
            if field not in signature.output_fields:
                raise ValueError(f"{field} must be in signature")

        if signature.output_fields["plot_type"].annotation is not str:
            raise ValueError("plot_type must be a string")
        if signature.output_fields["subplot_count"].annotation is not int:
            raise ValueError("subplot_count must be an integer")
        if (
            signature.output_fields["is_extractable_plot"].annotation
            is not bool
        ):
            raise ValueError("is_extractable must be a boolean")


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


def make_dspy_plot_information_extractor_signature(
    signature_name: str = "PlotIdentificationSignature",
    instructions: str | None = None,
    figure_base64_description: str | None = None,
    publication_context_description: str | None = None,
    plot_type_description: str | None = None,
    subplot_count_description: str | None = None,
    is_extractable_description: str | None = None,
) -> type[dspy.Signature]:
    """
    Create a dspy signature for extracting plot information from figures.

    Args:
        signature_name (str): Name of the signature.
        instructions (str): Instructions for the signature.
        figure_base64_description (str): Description for the base64
            image input.
        publication_context_description (str): Description for the publication
            context input.
        plot_type_description (str): Description for the plot type output.
        subplot_count_description (str): Description for the subplot count
            output.
        is_extractable_description (str): Description for the is_extractable
            output.

    Returns:
        dspy.Signature: The constructed dspy signature for plot information
            extraction.
    """
    signature = PlotIdentificationSignature
    if instructions is None:
        signature = signature.with_instructions(instructions)
    if figure_base64_description is None:
        signature = signature.with_updated_fields(
            "figure_base64",
            desc=figure_base64_description,
        )
    if publication_context_description is None:
        signature = signature.with_updated_fields(
            "publication_context",
            desc=publication_context_description,
        )
    if plot_type_description is None:
        signature = signature.with_updated_fields(
            "plot_type",
            desc=plot_type_description,
        )
    if subplot_count_description is None:
        signature = signature.with_updated_fields(
            "subplot_count",
            desc=subplot_count_description,
        )
    if is_extractable_description is None:
        signature = signature.with_updated_fields(
            "is_extractable_plot",
            desc=is_extractable_description,
        )
    signature.__name__ = signature_name
    return signature
