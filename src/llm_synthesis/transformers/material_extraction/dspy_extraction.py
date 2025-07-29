import dspy

from llm_synthesis.transformers.material_extraction.base import (
    MaterialExtractorInterface,
)


class DspyTextExtractor(MaterialExtractorInterface):
    """
    A text extractor that uses dspy to extract any arbitrary text
    from the publication text.
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

    def forward(self, input: str) -> str:
        """
        Extract text from the given str using the language model and signature.

        Args:
            input (str): The str from which to extract text.

        Returns:
            str: The extracted text from the str.
        """
        predict_kwargs = {"publication_text": input}
        with dspy.settings.context(
            lm=self.lm,
            adapter=dspy.adapters.JSONAdapter(),
        ):
            return dspy.ChainOfThought(self.signature)(
                **predict_kwargs
            ).__getattr__(next(iter(self.signature.output_fields.keys())))

    def _validate_signature(self, signature: type[dspy.Signature]):
        """
        Validate that the signature contains the required input
        and output fields with correct types.

        Args:
            signature (dspy.Signature): The signature to validate.

        Raises:
            ValueError: If any required field is missing or has the wrong type.
        """
        if "publication_text" not in signature.input_fields:
            raise ValueError("Publication text must be in signature")
        if signature.input_fields["publication_text"].annotation is not str:
            raise ValueError("Publication text must be a string")
        if len(signature.output_fields) != 1:
            raise ValueError("Only one output field is allowed")
        if next(iter(signature.output_fields.values())).annotation is not str:
            raise ValueError("Output field must be a string")


def make_dspy_text_extractor_signature(
    signature_name: str = "DspyTextExtractorSignature",
    instructions: str = "Extract the synthesis paragraph from the publication"
    " text.",
    input_description: str = "The publication text to extract the synthesis"
    " paragraph from.",
    output_name: str = "synthesis_paragraph",
    output_description: str = "The extracted synthesis paragraph.",
) -> type[dspy.Signature]:
    """
    Create a dspy signature for extracting text from publication text.

    Args:
        signature_name (str): Name of the signature.
        instructions (str): Instructions for the signature.
        input_description (str): Description for the publication text input.
        output_name (str): Name of the output field.
        output_description (str): Description for the output field.

    Returns:
        dspy.Signature: The constructed dspy signature for text extraction.
    """
    signature = {
        "publication_text": (
            str,
            dspy.InputField(description=input_description),
        ),
        output_name: (str, dspy.OutputField(description=output_description)),
    }
    return dspy.make_signature(
        signature_name=signature_name,
        instructions=instructions,
        signature=signature,
    )
