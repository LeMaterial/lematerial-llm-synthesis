import json

import dspy

from llm_synthesis.models.figure import FigureInfoWithPaper
from llm_synthesis.models.plot import ExtractedPlotData
from llm_synthesis.transformers.plot_extraction.base import (
    PlotAnalysisSignature,
)
from llm_synthesis.utils.figure_utils import clean_text_from_images


class PlotAnalysisExtractor(PlotAnalysisSignature):
    """
    Extractor to analyze extracted plot data and provide insights.
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
        self, input: tuple[FigureInfoWithPaper, list[ExtractedPlotData]]
    ) -> str:
        """
        Analyze plot data using the language model and signature.

        Args:
            input (tuple[FigureInfoWithPaper, list[ExtractedPlotData]]):
            The figure and paper context, along with the extracted plot data.

        Returns:
            str: The analysis of the plot data.
        """
        figure_info, plot_data_list = input
        predict_kwargs = {
            "extracted_plot_data": json.dumps(
                [plot_data.model_dump() for plot_data in plot_data_list],
                indent=2,
                default=str,
            ),
            "publication_context": clean_text_from_images(
                figure_info.paper_text
            ),
            "figure_caption": clean_text_from_images(
                figure_info.context_before + figure_info.context_after
            ),
        }
        with dspy.settings.context(
            lm=self.lm, adapter=dspy.adapters.JSONAdapter()
        ):
            result = dspy.ChainOfThought(self.signature)(**predict_kwargs)
            return result.scientific_analysis

    def _validate_signature(self, signature: type[dspy.Signature]):
        """
        Validate that the signature contains all required input and output
        fields with correct types.

        Args:
            signature (dspy.Signature): The signature to validate.

        Raises:
            ValueError: If required field is missing or has the wrong type.
        """
        required_input_fields = [
            "extracted_plot_data",
            "publication_context",
            "figure_caption",
        ]

        for field in required_input_fields:
            if field not in signature.input_fields:
                raise ValueError(f"{field} must be in signature")
            if signature.input_fields[field].annotation is not str:
                raise ValueError(f"{field} must be a string")

        if "scientific_analysis" not in signature.output_fields:
            raise ValueError("scientific_analysis must be in signature")
        if signature.output_fields["scientific_analysis"].annotation is not str:
            raise ValueError("scientific_analysis must be a string")


class PlotAnalysisSignature(dspy.Signature):
    """
    Signature for detailed scientific analysis of extracted plot data
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


def make_dspy_plot_analysis_extractor_signature(
    signature_name: str = "PlotAnalysisSignature",
    instructions: str | None = None,
    extracted_plot_data_description: str | None = None,
    publication_context_description: str | None = None,
    figure_caption_description: str | None = None,
    scientific_analysis_description: str | None = None,
) -> type[dspy.Signature]:
    """
    Create a dspy signature for analyzing plot data.

    Args:
        signature_name (str): Name of the signature.
        instructions (str): Instructions for the signature.
        extracted_plot_data_description (str): Desc. extracted plot data input.
        publication_context_description (str): Desc. publication context input.
        figure_caption_description (str): Description of figure caption input.
        scientific_analysis_description (str): Desc. sc. analysis output.

    Returns:
        dspy.Signature: The constructed dspy signature for plot analysis.
    """
    signature = PlotAnalysisSignature
    if instructions is not None:
        signature = signature.with_instructions(instructions)
    if extracted_plot_data_description is not None:
        signature = signature.with_updated_fields(
            "extracted_plot_data",
            desc=extracted_plot_data_description,
        )
    if publication_context_description is not None:
        signature = signature.with_updated_fields(
            "publication_context",
            desc=publication_context_description,
        )
    if figure_caption_description is not None:
        signature = signature.with_updated_fields(
            "figure_caption",
            desc=figure_caption_description,
        )
    if scientific_analysis_description is not None:
        signature = signature.with_updated_fields(
            "scientific_analysis",
            desc=scientific_analysis_description,
        )
    signature.__name__ = signature_name
    return signature
