import dspy
from llm_synthesis.extraction.synthesis.signatures import StructuredSynthesisSignature
from llm_synthesis.ontologies import GeneralSynthesisOntology


class StructuredSynthesisParser(dspy.Module):
    def __init__(self, signature: dspy.Signature = StructuredSynthesisSignature):
        self.predict = dspy.Predict(signature)
        # TODO: add more elaborate components here: CoT, Typed CoT, multi-hop reasoning, ...
        # Do this modularly in order to switch up easily for benchmark

    def __call__(self, synthesis_procedure: str) -> GeneralSynthesisOntology:
        return self.predict(synthesis_procedure=synthesis_procedure)
