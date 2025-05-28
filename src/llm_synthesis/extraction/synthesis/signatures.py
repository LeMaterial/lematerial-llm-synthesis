import dspy
from llm_synthesis.ontologies import GeneralSynthesisOntology


class StructuredSynthesisSignature(dspy.Signature):
    """Signature for the structured synthesis parser."""

    synthesis_procedure: str = dspy.InputField(
        description="The synthesis procedure to parse."
    )
    structured_synthesis_procedure: GeneralSynthesisOntology = dspy.OutputField(
        description="The structured synthesis procedure."
    )


class SynthesisSignature(dspy.Signature):
    """Signature for the synthesis parser."""

    synthesis_procedure: str = dspy.InputField(
        description="The synthesis procedure to parse."
    )
    structured_synthesis_procedure: str = dspy.OutputField(
        description="The structured synthesis procedure."
    )


class CleanupSignature(dspy.Signature):
    """Clean context before predicting the synthesis procedure."""

    publication_text: str = dspy.InputField(
        description="The whole publication text including redundant information."
    )
    synthesis_procedure: str = dspy.OutputField(
        description="The synthesis procedure of all compounds synthesized in the publication. Includes information from the methods, introduction, appendix."
    )
