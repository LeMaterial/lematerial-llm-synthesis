from llm_synthesis.transformers.synthesis_extraction import (
    dspy_synthesis_extraction,
)

DspyStructuredSynthesisExtractor = (
    dspy_synthesis_extraction.DspyStructuredSynthesisExtractor
)
make_dspy_structured_synthesis_extractor_signature = (
    dspy_synthesis_extraction.make_dspy_structured_synthesis_extractor_signature
)

__all__ = [
    "DspyStructuredSynthesisExtractor",
    "make_dspy_structured_synthesis_extractor_signature",
]
