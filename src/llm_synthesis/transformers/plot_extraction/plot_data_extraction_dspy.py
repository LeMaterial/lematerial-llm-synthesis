import dspy

from llm_synthesis.models.figure import FigureInfoWithPaper
from llm_synthesis.models.plot import (
    ExtractedPlotData,
    PlotMetadata,
)
from llm_synthesis.transformers.plot_extraction.base import (
    PlotDataExtractorInterface,
)
from llm_synthesis.utils.figure_utils import clean_text_from_images


class PlotDataExtractor(PlotDataExtractorInterface):
    """
    Extractor that uses dspy to extract plot data from figures.
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

    def forward(
        self, input: tuple[FigureInfoWithPaper, str]
    ) -> list[ExtractedPlotData]:
        """
        Extract plot data using the language model and signature.

        Args:
            input (tuple[FigureInfoWithPaper, str]): The figure and paper
            context, along with subplot focus information.

        Returns:
            list[ExtractedPlotData]: The extracted plot data.
        """
        figure_info, subplot_focus = input
        predict_kwargs = {
            "figure_base64": figure_info.base64_data,
            "publication_context": clean_text_from_images(
                figure_info.paper_text
            ),
            "subplot_focus": subplot_focus,
        }
        with dspy.settings.context(
            lm=self.lm, adapter=dspy.adapters.JSONAdapter()
        ):
            result = dspy.ChainOfThought(self.signature)(**predict_kwargs)

            # The signature now returns a list of ExtractedPlotData
            # For now, we'll take the first one or create a default
            if result.extracted_data and len(result.extracted_data) > 0:
                return result.extracted_data
            else:
                # Return empty data if no extraction was possible
                return [
                    ExtractedPlotData(
                        metadata=PlotMetadata(
                            x_axis_label="",
                            left_y_axis_label="",
                            is_dual_axis=False,
                        ),
                        data_series=[],
                        technical_takeaways=[],
                    )
                ]

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
            "subplot_focus",
        ]

        for field in required_input_fields:
            if field not in signature.input_fields:
                raise ValueError(f"{field} must be in signature")
            if signature.input_fields[field].annotation is not str:
                raise ValueError(f"{field} must be a string")

        if "extracted_data" not in signature.output_fields:
            raise ValueError("extracted_data must be in signature")
        # Note: We can't easily validate the list type annotation here
        # The actual validation will happen at runtime


class DataExtractionSignature(dspy.Signature):
    """
    Signature for extracting actual data points from identified plots.
    """

    figure_base64: str = dspy.InputField(
        description="Base64 encoded image of the extractable scientific plot"
    )
    publication_context: str = dspy.InputField(
        description="Relevant publication text providing context about the"
        " figure"
    )
    subplot_focus: str = dspy.InputField(
        description="Which subplot to focus on if multiple (e.g.,"
        " 'subplot a', 'all subplots', 'main plot')"
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


def make_dspy_plot_data_extractor_signature(
    signature_name: str = "DataExtractionSignature",
    instructions: str | None = None,
    figure_base64_description: str | None = None,
    publication_context_description: str | None = None,
    subplot_focus_description: str | None = None,
    extracted_data_description: str | None = None,
) -> type[dspy.Signature]:
    """
    Create a dspy signature for extracting plot data from figures.

    Args:
        signature_name (str): Name of the signature.
        instructions (str): Instructions for the signature.
        figure_base64_description (str): Description for the base64
            image input.
        publication_context_description (str): Description for the publication
            context input.
        subplot_focus_description (str): Description for the subplot focus
            input.
        extracted_data_description (str): Description for the extracted data
            output.

    Returns:
        dspy.Signature: The constructed dspy signature for plot data extraction
    """
    signature = DataExtractionSignature
    if instructions is not None:
        signature = signature.with_instructions(instructions)
    if figure_base64_description is not None:
        signature = signature.with_updated_fields(
            "figure_base64",
            desc=figure_base64_description,
        )
    if publication_context_description is not None:
        signature = signature.with_updated_fields(
            "publication_context",
            desc=publication_context_description,
        )
    if subplot_focus_description is not None:
        signature = signature.with_updated_fields(
            "subplot_focus",
            desc=subplot_focus_description,
        )
    if extracted_data_description is not None:
        signature = signature.with_updated_fields(
            "extracted_data",
            desc=extracted_data_description,
        )
    signature.__name__ = signature_name
    return signature
