import dspy
import logging
# ADDED: Import JSON repair utilities to handle malformed LLM outputs
# These functions provide robust parsing with multiple fallback strategies
from llm_synthesis.utils.json_utils import parse_json, extract_json, validate_required_fields

from llm_synthesis.models.ontologies.general import GeneralSynthesisOntology
from llm_synthesis.transformers.synthesis_extraction.base import (
    StructuredSynthesisExtractorInterface,
)

# ADDED: Logger for tracking JSON repair operations and debugging
logger = logging.getLogger(__name__)

class DspyStructuredSynthesisExtractor(StructuredSynthesisExtractorInterface):
    """
    Extractor that uses dspy to extract a structured synthesis ontology
    from a synthesis paragraph.
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
            input (str): The synthesis paragraph to process.

        Returns:
            GeneralSynthesisOntology: The structured synthesis ontology
            extracted from the paragraph.
        """
        predict_kwargs = {"synthesis_paragraph": input}
        with dspy.settings.context(lm=self.lm):
            # ENHANCED: Changed from direct return to variable assignment
            # Store result to enable JSON repair processing
            result = dspy.Predict(self.signature, lm=self.lm)(
                **predict_kwargs
            )
            # JSON repair enhancement: handle malformed LLM output
            # ADDED: Extract raw JSON string from DSPy result for processing
            # Using list(result.values())[0] is more reliable than complex __getattr__ chains
            # This handles any DSPy signature structure consistently
            json_str = list(result.values())[0]
            
            # Handle non-scientific content
            if json_str.strip() == "NON_SCIENTIFIC_FIGURE":
                return GeneralSynthesisOntology(
                    target_compound="NON_SCIENTIFIC_CONTENT",
                    starting_materials=[],
                    steps=[],
                    notes="Non-scientific content - no synthesis procedure available"
                )
            # Parse with centralized JSON repair
            # ADDED: Apply primary JSON repair strategy with centralized parsing
            # parse_json handles: incomplete JSON, missing quotes, malformed structures
            # fallback_value={} ensures we always get a dict for further processing
            json_obj = parse_json(json_str, fallback_value={})
            
            # ADDED: Secondary repair strategy if primary parsing fails
            # extract_json uses different algorithms to find JSON within text
            # This catches cases where JSON is embedded in explanatory text
            if not validate_required_fields(json_obj, ['target_compound', 'starting_materials', 'steps']):
                json_obj = extract_json(json_str, fallback_value={})
            
            # Convert to ontology or create fallback
            if validate_required_fields(json_obj, ['target_compound', 'starting_materials', 'steps']):
                return GeneralSynthesisOntology(
                    target_compound=json_obj.get('target_compound', ''),
                    # Handle both 'starting_materials' and 'materials' field names
                    starting_materials=json_obj.get('starting_materials', json_obj.get('materials', [])),
                    steps=json_obj.get('steps', []),
                    notes=json_obj.get('notes', '')
                )
            else:
                # Fallback ontology
                return GeneralSynthesisOntology(
                    target_compound="EXTRACTION_FAILED",
                    starting_materials=[],
                    steps=[],
                    notes=f"Failed to extract synthesis information: {json_str[:200]}..."
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
