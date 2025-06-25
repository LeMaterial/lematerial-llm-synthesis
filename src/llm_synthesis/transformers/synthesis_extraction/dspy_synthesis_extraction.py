import dspy

from llm_synthesis.models.ontologies import GeneralSynthesisOntology
from llm_synthesis.transformers.synthesis_extraction.base import (
    StructuredSynthesisExtractorInterface,
)


class DspyStructuredSynthesisExtractor(StructuredSynthesisExtractorInterface):
    """
    Extractor that uses dspy to extract a structured synthesis ontology
    from a publication.
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

    def forward(self, input: str) -> GeneralSynthesisOntology:
        """
            Extract a structured synthesis ontology from the given synthesis
            paragraph using the language model and signature.

        Args:
            input (str): The publication text to process.

        Returns:
            GeneralSynthesisOntology: The structured synthesis ontology
                                      extracted from the publication text.
        """
        predict_kwargs = {"publication_text": input}
        with dspy.settings.context(lm=self.lm):
            return dspy.Predict(self.signature, lm=self.lm)(
                **predict_kwargs
            ).__getattr__(next(iter(self.signature.output_fields.keys())))

    def _validate_signature(self, signature: type[dspy.Signature]):
        """
        Validate that the signature contains the required input and output
        fields with correct types.

        Args:
            signature (dspy.Signature): The signature to validate.

        Raises:
            ValueError: If any required field is missing or has the wrong type.
        """
        if "synthesis_paragraph" not in signature.input_fields:
            raise ValueError("Synthesis paragraph must be in signature")
        if signature.input_fields["synthesis_paragraph"].annotation is not str:
            raise ValueError("Synthesis paragraph must be a string")
        if len(signature.output_fields) != 1:
            raise ValueError("Only one output field is allowed")
        if (
            next(iter(signature.output_fields.values())).annotation
            is not GeneralSynthesisOntology
        ):
            raise ValueError("Output field must be a GeneralSynthesisOntology")


def make_dspy_structured_synthesis_extractor_signature(
    signature_name: str = "DspyStructuredSynthesisExtractorSignature",
    instructions: str = "Extract the structured synthesis from the synthesis "
    "paragraph.",
    input_description: str = "The synthesis paragraph to extract the"
    " structured synthesis from.",
    output_name: str = "structured_synthesis",
    output_description: str = "The extracted structured synthesis.",
) -> type[dspy.Signature]:
    """
    Create a dspy signature for extracting a structured synthesis ontology from
    a synthesis paragraph.

    Args:
        signature_name (str): Name of the signature.
        instructions (str): Instructions for the signature.
        input_description (str): Description for the synthesis paragraph input.
        output_name (str): Name of the output field.
        output_description (str): Description for the output field.

    Returns:
        dspy.Signature: The constructed dspy signature for structured synthesis
                        extraction.
    """
    signature = {
        "synthesis_paragraph": (
            str,
            dspy.InputField(description=input_description),
        ),
        output_name: (
            GeneralSynthesisOntology,
            dspy.OutputField(description=output_description),
        ),
    }
    return dspy.make_signature(
        signature_name=signature_name,
        instructions=instructions,
        signature=signature,
    )
