import dspy

from llm_synthesis.models.figure import FigureInfoWithPaper
from llm_synthesis.transformers.figure_description.base import (
    FigureDescriptionExtractorInterface,
)


class DspyFigureDescriptionExtractor(FigureDescriptionExtractorInterface):
    """
    Extractor that uses dspy to generate a figure description from figure and paper context.
    """

    def __init__(self, signature: dspy.Signature, lm: dspy.LM):
        """
        Initialize the extractor with a dspy signature and language model.

        Args:
            signature (dspy.Signature): The dspy signature specifying input/output fields.
            lm (dspy.LM): The language model to use for prediction.
        """
        self._validate_signature(signature)
        self.signature = signature
        self.lm = lm

    def extract(self, input: FigureInfoWithPaper) -> str:
        """
        Extract a figure description using the language model and signature.

        Args:
            input (FigureInfoWithPaper): The figure and paper context.

        Returns:
            str: The generated figure description.
        """
        predict_kwargs = {
            "publication_text": input.paper_text,
            "si_text": input.si_text,
            "figure_base64": input.base64_data,
            "caption_context": input.context_before + input.context_after,
            "figure_position_info": input.figure_reference,
        }
        with dspy.settings.context(lm=self.lm):
            return dspy.Predict(self.signature)(**predict_kwargs).__getattr__(
                list(self.signature.output_fields.keys())[0]
            )

    def _validate_signature(self, signature: dspy.Signature):
        """
        Validate that the signature contains all required input and output fields with correct types.

        Args:
            signature (dspy.Signature): The signature to validate.

        Raises:
            ValueError: If any required field is missing or has the wrong type.
        """
        if "publication_text" not in signature.input_fields:
            raise ValueError("Publication text must be in signature")
        if signature.input_fields["publication_text"].annotation is not str:
            raise ValueError("Publication text must be a string")
        if "si_text" not in signature.input_fields:
            raise ValueError("SI text must be in signature")
        if signature.input_fields["si_text"].annotation is not str:
            raise ValueError("SI text must be a string")
        if "figure_base64" not in signature.input_fields:
            raise ValueError("Figure base64 must be in signature")
        if signature.input_fields["figure_base64"].annotation is not str:
            raise ValueError("Figure base64 must be a string")
        if "caption_context" not in signature.input_fields:
            raise ValueError("Caption context must be in signature")
        if signature.input_fields["caption_context"].annotation is not str:
            raise ValueError("Caption context must be a string")
        if "figure_position_info" not in signature.input_fields:
            raise ValueError("Figure position info must be in signature")
        if signature.input_fields["figure_position_info"].annotation is not str:
            raise ValueError("Figure position info must be a string")
        if "figure_description" not in signature.output_fields:
            raise ValueError("Figure description must be in signature")
        if signature.output_fields["figure_description"].annotation is not str:
            raise ValueError("Figure description must be a string")


def make_dspy_figure_description_extractor_signature(
    signature_name: str = "DspyFigureDescriptionExtractorSignature",
    instructions: str = "Extract the figure description from the figure.",
    publication_text_description: str = "The publication text to extract the figure description from.",
    si_text_description: str = "The supporting information text to extract the figure description from.",
    figure_base64_description: str = "The base64 encoded image of the figure to extract the description from.",
    caption_context_description: str = "The text context surrounding the figure position including the figure caption and nearby paragraphs that reference this figure.",
    figure_position_info_description: str = "The information about the figure's position in the document (e.g., 'Figure 2', 'Fig. 3a', 'Scheme 1') to help with contextual understanding.",
    figure_description_description: str = "The extracted figure description.",
) -> dspy.Signature:
    """
    Create a dspy signature for extracting figure descriptions.

    Args:
        signature_name (str): Name of the signature.
        instructions (str): Instructions for the signature.
        publication_text_description (str): Description for the publication text input.
        si_text_description (str): Description for the SI text input.
        figure_base64_description (str): Description for the base64 image input.
        caption_context_description (str): Description for the caption context input.
        figure_position_info_description (str): Description for the figure position info input.
        figure_description_description (str): Description for the output figure description.

    Returns:
        dspy.Signature: The constructed dspy signature for figure description extraction.
    """
    signature = {
        "publication_text": (
            str,
            dspy.InputField(description=publication_text_description),
        ),
        "si_text": (str, dspy.InputField(description=si_text_description)),
        "figure_base64": (str, dspy.InputField(description=figure_base64_description)),
        "caption_context": (
            str,
            dspy.InputField(description=caption_context_description),
        ),
        "figure_position_info": (
            str,
            dspy.InputField(description=figure_position_info_description),
        ),
        "figure_description": (
            str,
            dspy.OutputField(description=figure_description_description),
        ),
    }
    return dspy.make_signature(
        signature_name=signature_name,
        instructions=instructions,
        signature=signature,
    )
