import copy
import logging

import dspy

from llm_synthesis.transformers.material_extraction.base import (
    MaterialExtractorInterface,
)

log = logging.getLogger(__name__)


class DspyTextExtractor(MaterialExtractorInterface):
    """
    A text extractor that uses dspy to extract any arbitrary text
    from the publication text.

    Implements temperature escalation retry on failures to improve
    robustness against transient validation errors.
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

    def forward(self, input: str) -> str:
        """
        Extract text from the given str using the language model and signature.

        Retries at escalating temperatures on failure.

        Args:
            input (str): The str from which to extract text.

        Returns:
            str: The extracted text from the str.
        """
        predict_kwargs = {"publication_text": input}
        output_field = next(iter(self.signature.output_fields.keys()))
        last_exc: Exception | None = None

        for t_idx, temp in enumerate(self.retry_temperatures):
            lm = self._lm_with_overrides({"temperature": temp})
            try:
                with dspy.settings.context(
                    lm=lm,
                    adapter=dspy.adapters.JSONAdapter(),
                ):
                    return dspy.ChainOfThought(self.signature)(
                        **predict_kwargs
                    ).__getattr__(output_field)
            except Exception as e:
                last_exc = e
                if t_idx < len(self.retry_temperatures) - 1:
                    log.warning(
                        "Material extractor: failure at temp=%.1f: %r"
                        " — retrying at temp=%.1f",
                        temp,
                        e,
                        self.retry_temperatures[t_idx + 1],
                    )
                else:
                    log.warning(
                        "Material extractor: all temperatures exhausted: %r",
                        e,
                    )

        if last_exc is None:
            # Defensive: the loop body always sets last_exc on failure and
            # returns on success, so this branch is unreachable in practice.
            raise RuntimeError(
                "material extraction failed without recording an exception"
            )
        raise last_exc

    def _lm_with_overrides(self, overrides: dict) -> dspy.LM:
        """Return a shallow copy of self.lm with kwargs overridden."""
        lm = copy.copy(self.lm)
        lm.kwargs = {**self.lm.kwargs, **overrides}
        return lm

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
