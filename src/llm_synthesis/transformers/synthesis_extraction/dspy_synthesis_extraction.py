import logging

import dspy

from llm_synthesis.models.ontologies import GeneralSynthesisOntology
from llm_synthesis.transformers.synthesis_extraction.base import (
    SynthesisExtractorInterface,
)


class DspySynthesisExtractor(SynthesisExtractorInterface):
    """
    Extractor that uses dspy to extract a structured synthesis ontology
    for a specific material from the entire paper text.
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

    def forward(self, input: tuple[str, str]) -> GeneralSynthesisOntology:
        """
        Extract a structured synthesis ontology for a specific material
        from the given paper text.

        Args:
            input (tuple[str, str]): Tuple of (paper_text, material_name).

        Returns:
            GeneralSynthesisOntology: The structured synthesis ontology
                                      for the specific material.
        """
        paper_text, material_name = input
        predict_kwargs = {
            "paper_text": paper_text,
            "material_name": material_name,
        }

        try:
            with dspy.settings.context(lm=self.lm):
                result = dspy.Predict(self.signature, lm=self.lm)(
                    **predict_kwargs
                )
                return result.__getattr__(
                    next(iter(self.signature.output_fields.keys()))
                )
        except Exception as e:
            # Fallback: create a minimal synthesis ontology if extraction fails
            logging.warning(
                f"Failed to extract synthesis for {material_name}: {e}"
            )
            return GeneralSynthesisOntology(
                synthesis_id=None,
                target_compound=material_name,
                synthesis_method=None,
                starting_materials=[],
                steps=[],
                major_equipment=[],
                characterization_methods=[],
                notes=f"Extraction failed: {e!s}",
            )

    def _validate_signature(self, signature: type[dspy.Signature]):
        """
        Validate that the signature contains the required input and output
        fields with correct types.

        Args:
            signature (dspy.Signature): The signature to validate.

        Raises:
            ValueError: If any required field is missing or has the wrong type.
        """
        if "paper_text" not in signature.input_fields:
            raise ValueError("Paper text must be in signature")
        if signature.input_fields["paper_text"].annotation is not str:
            raise ValueError("Paper text must be a string")
        if "material_name" not in signature.input_fields:
            raise ValueError("Material name must be in signature")
        if signature.input_fields["material_name"].annotation is not str:
            raise ValueError("Material name must be a string")
        if len(signature.output_fields) != 1:
            raise ValueError("Only one output field is allowed")
        if (
            next(iter(signature.output_fields.values())).annotation
            is not GeneralSynthesisOntology
        ):
            raise ValueError("Output field must be a GeneralSynthesisOntology")


def make_dspy_synthesis_extractor_signature(
    signature_name: str = "DspySynthesisExtractorSignature",
    instructions: str = (
        "Extract structured synthesis for a specific material from the paper. "
        "Output only a valid JSON with the structured_synthesis field."
    ),
    paper_text_description: str = (
        "Complete paper text to search for the material synthesis procedure."
    ),
    material_name_description: str = (
        "The name of the specific material to extract synthesis for."
    ),
    output_name: str = "structured_synthesis",
    output_description: str = (
        "The extracted structured synthesis for specific material as a JSON."
    ),
) -> type[dspy.Signature]:
    """
    Create signature for extracting a materials-specific synthesis ontology.

    Args:
        signature_name (str): Name of the signature.
        instructions (str): Instructions for the signature.
        paper_text_description (str): Description for the paper text input.
        material_name_description (str): Description for material name input.
        output_name (str): Name of the output field.
        output_description (str): Description for the output field.

    Returns:
        dspy.Signature: The dspy signature for synthesis extraction.
    """
    signature = {
        "paper_text": (
            str,
            dspy.InputField(description=paper_text_description),
        ),
        "material_name": (
            str,
            dspy.InputField(description=material_name_description),
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
