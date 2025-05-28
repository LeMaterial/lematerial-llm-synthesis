import dspy
from llm_synthesis.extraction.synthesis.signatures import (
    SynthesisSignature,
    CleanupSignature,
)


class MultiLayerSynthesisParser(dspy.Module):
    def __init__(
        self,
        cleanup_signature: dspy.Signature = CleanupSignature,
        synthesis_signature: dspy.Signature = SynthesisSignature,
    ):
        self.cleanup = dspy.Predict(cleanup_signature)
        self.predict = dspy.Predict(synthesis_signature)
        # TODO: add more elaborate components here: CoT, Typed CoT, multi-hop reasoning, ...
        # Do this modularly in order to switch up easily for benchmark

    def __call__(self, synthesis_procedure: str) -> str:
        return self.predict(synthesis_procedure=synthesis_procedure)
