import copy
import json
import logging
import re

import dspy

from llm_synthesis.models.ontologies import GeneralSynthesisOntology
from llm_synthesis.transformers.synthesis_extraction.base import (
    SynthesisExtractorInterface,
)


class SynthesisJSONAdapter(dspy.adapters.JSONAdapter):
    """Custom adapter for handling synthesis extraction with JSON wrapper."""

    def __init__(self):
        super().__init__()

    def extract(self, response: str, signature: type[dspy.Signature]) -> dict:
        """Extract structured data from response."""
        try:
            # First try the standard JSON adapter
            return super().extract(response, signature)
        except Exception as e:
            # Try to parse the JSON and extract structured_synthesis
            try:
                parsed = json.loads(response)
                if "structured_synthesis" in parsed:
                    # Return the structured_synthesis content directly
                    return {
                        "structured_synthesis": parsed["structured_synthesis"]
                    }
                else:
                    # If no wrapper, assume the response is the direct content
                    return {"structured_synthesis": parsed}
            except Exception as json_error:
                logging.debug(f"Failed to parse JSON response: {json_error}")
                raise e


class DspySynthesisExtractor(SynthesisExtractorInterface):
    """
    Extractor that uses dspy to extract a structured synthesis ontology
    for a specific material from the entire paper text.

    Implements temperature escalation retry and bare-JSON recovery on
    failures to improve robustness without changing the happy-path behavior.
    """

    def __init__(
        self,
        signature: type[dspy.Signature],
        lm: dspy.LM,
        retry_temperatures: list[float] | None = None,
    ):
        """
        Initialize the extractor with a dspy signature and language model.

        Args:
            signature (dspy.Signature): The dspy signature specifying
                                        input/output fields.
            lm (dspy.LM): The language model to use for prediction.
            retry_temperatures: Temperatures to try on failures.
                                 Defaults to [0.0, 0.3, 0.5].
        """
        self._validate_signature(signature)
        self.signature = signature
        self.retry_temperatures = retry_temperatures or [0.0, 0.3, 0.5]

        # Strip any pre-existing response_format / extra_body so retries
        # have full control over LM kwargs.
        _clean_keys = {"response_format", "extra_body"}
        if _clean_keys & set(lm.kwargs):
            self.lm = copy.copy(lm)
            self.lm.kwargs = {
                k: v for k, v in lm.kwargs.items() if k not in _clean_keys
            }
        else:
            self.lm = lm

    def forward(self, input: tuple[str, str]) -> GeneralSynthesisOntology:
        """
        Extract a structured synthesis ontology for a specific material
        from the given paper text.

        Retries at escalating temperatures on failure. Attempts bare-JSON
        recovery before escalating. Falls back to a minimal ontology if all
        attempts are exhausted.

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

        last_exc: Exception | None = None

        for t_idx, temp in enumerate(self.retry_temperatures):
            lm = self._lm_with_overrides({"temperature": temp})
            try:
                with dspy.settings.context(
                    lm=lm, adapter=SynthesisJSONAdapter()
                ):
                    result = dspy.Predict(self.signature)(**predict_kwargs)
                    synthesis_data = result.__getattr__(
                        next(iter(self.signature.output_fields.keys()))
                    )

                    # Ensure required fields are present
                    if (
                        not hasattr(synthesis_data, "target_compound_type")
                        or synthesis_data.target_compound_type is None
                    ):
                        synthesis_data.target_compound_type = "other"
                    if (
                        not hasattr(synthesis_data, "synthesis_method")
                        or synthesis_data.synthesis_method is None
                    ):
                        synthesis_data.synthesis_method = "other"

                    return synthesis_data

            except Exception as e:
                # Attempt bare-JSON recovery before escalating temperature.
                recovered = self._try_recover_bare_json(e, material_name)
                if recovered is not None:
                    logging.info(
                        "Synthesis extractor: recovered bare JSON for %s"
                        " (temp=%.1f)",
                        material_name,
                        temp,
                    )
                    return recovered

                last_exc = e
                if t_idx < len(self.retry_temperatures) - 1:
                    logging.warning(
                        "Synthesis extractor: failure at temp=%.1f for %s:"
                        " %r — retrying at temp=%.1f",
                        temp,
                        material_name,
                        e,
                        self.retry_temperatures[t_idx + 1],
                    )
                else:
                    logging.warning(
                        "Synthesis extractor: all temperatures exhausted"
                        " for %s: %r",
                        material_name,
                        e,
                    )

        # Try to parse raw response as JSON and extract structured_synthesis
        try:
            # Get the raw response from the LM
            raw_response = (
                self.lm.history[-1]["response"]
                if hasattr(self.lm, "history") and self.lm.history
                else None
            )
            if raw_response:
                # Try to parse as JSON
                parsed = json.loads(raw_response)
                if "structured_synthesis" in parsed:
                    synthesis_data = parsed["structured_synthesis"]
                    # Ensure required fields are present
                    if "target_compound_type" not in synthesis_data:
                        synthesis_data["target_compound_type"] = "other"
                    if (
                        "synthesis_method" not in synthesis_data
                        or synthesis_data["synthesis_method"] is None
                    ):
                        synthesis_data["synthesis_method"] = "other"
                    return GeneralSynthesisOntology(**synthesis_data)
        except Exception as json_error:
            logging.debug(f"Failed to parse JSON response: {json_error}")

        # Fallback: create a minimal synthesis ontology if extraction fails
        logging.warning(
            f"Failed to extract synthesis for {material_name}: {last_exc}"
        )
        return GeneralSynthesisOntology(
            target_compound=material_name,
            target_compound_type="other",
            synthesis_method="other",
            starting_materials=[],
            steps=[],
            equipment=[],
            notes=f"Extraction failed: {last_exc!s}",
        )

    # ── Private helpers ────────────────────────────────────────────────────

    def _lm_with_overrides(self, overrides: dict) -> dspy.LM:
        """Return a shallow copy of self.lm with kwargs overridden."""
        lm = copy.copy(self.lm)
        lm.kwargs = {**self.lm.kwargs, **overrides}
        return lm

    @staticmethod
    def _try_recover_bare_json(
        exc: Exception,
        material_name: str,
    ) -> GeneralSynthesisOntology | None:
        """
        Attempt to recover when the model returned a complete, valid
        GeneralSynthesisOntology JSON but without the DSPy-expected
        ``{"structured_synthesis": {...}}`` wrapper key.

        DSPy's JSONAdapter embeds the raw LM response in the
        AdapterParseError message, so we can parse it directly from
        the exception.

        Returns the parsed ontology on success, None otherwise.
        """
        msg = str(exc)
        match = re.search(r"LM Response:\s*(\{)", msg, re.DOTALL)
        if not match:
            return None

        try:
            data, _ = json.JSONDecoder().raw_decode(msg, match.start(1))
        except json.JSONDecodeError:
            return None

        # Already wrapped — SynthesisJSONAdapter should have handled this
        if "structured_synthesis" in data:
            return None

        # Ensure required fields before validating
        data.setdefault("target_compound", material_name)
        data.setdefault("target_compound_type", "other")
        if not data.get("synthesis_method"):
            data["synthesis_method"] = "other"

        try:
            return GeneralSynthesisOntology.model_validate(data)
        except Exception:
            return None

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
