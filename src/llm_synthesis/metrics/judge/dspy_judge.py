import dspy

from llm_synthesis.metrics.judge.base import SynthesisJudgeInterface
from llm_synthesis.metrics.judge.evaluation_ontology import SynthesisEvaluation


class DspySynthesisJudge(SynthesisJudgeInterface):
    """
    A DSPy module that acts as an expert judge to evaluate LLM extracted
    synthesis recipes against a reference synthesis procedure.
    """

    def __init__(self, signature: type[dspy.Signature], lm: dspy.LM):
        self.signature = signature
        self.lm = lm
        super().__init__()

    def forward(self, input: tuple[str, str, str]) -> SynthesisEvaluation:
        target_material, extracted_recipe, synthesis_procedure = input
        with dspy.settings.context(lm=self.lm):
            prediction = dspy.Predict(self.signature)(
                target_material=target_material,
                extracted_recipe=extracted_recipe,
                synthesis_procedure=synthesis_procedure,
            )
            return prediction.evaluation


class SynthesisJudgeSignature(dspy.Signature):
    """
    You are an expert materials scientist tasked with evaluating an
    LLM-extracted synthesis recipe against a reference synthesis procedure.

    Evaluate the extracted recipe according to the following criteria on a
    scale of 1-5 (1.0: poor, 5.0: excellent, 0.5 step increments):

    1. Materials Appropriateness: Are the selected materials suitable for the
       target synthesis?
    2. Equipment Appropriateness: Is the selected equipment suitable for the
       synthesis?
    3. Procedure Completeness: Are all necessary steps included with
       sufficient detail?
    4. Procedure Similarity: How closely does the procedure match the
       reference?
    5. Procedure Feasibility: Can this procedure be realistically executed in
       a lab?
    6. Characterization Appropriateness: Are the characterization metrics
       suitable for validating success?
    7. Characterization Similarity: How well do predicted properties match
       the reference results?

    Provide detailed reasoning for each individual score, then calculate an
    overall score as the average of all criteria.

    Your response must include:
    - High-level reasoning overview
    - Individual reasoning for each of the 7 criteria
    - Numerical scores for each criterion and overall
    """

    target_material: str = dspy.InputField(
        description="Description of the target material to be synthesized"
    )
    extracted_recipe: str = dspy.InputField(
        description="The synthesis recipe extracted by an LLM"
    )
    synthesis_procedure: str = dspy.InputField(
        description=(
            "Reference synthesis procedure from scientific literature"
        )
    )
    evaluation: SynthesisEvaluation = dspy.OutputField(
        description="Structured evaluation with reasoning and scores"
    )


def make_dspy_synthesis_judge_signature(
    signature_name: str = "DspySynthesisJudgeSignature",
    instructions: str = (
        "Evaluate an LLM-extracted synthesis recipe against a reference "
        "synthesis procedure from the perspective of an expert materials "
        "scientist."
    ),
    target_material_description: str = (
        "A description of the target material to be synthesized."
    ),
    extracted_recipe_description: str = "The extracted synthesis recipe.",
    synthesis_procedure_description: str = (
        "The reference synthesis procedure from a scientific publication."
    ),
    evaluation_description: str = (
        "The structured evaluation containing detailed reasoning and scores."
    ),
) -> type[dspy.Signature]:
    """
    Create a dspy signature for evaluating synthesis recipes.

    Args:
        signature_name (str): Name of the signature.
        instructions (str): Instructions for the signature.
        target_material_description (str): Description for the target material
            input.
        extracted_recipe_description (str): Description for the extracted
            recipe input.
        synthesis_procedure_description (str): Description for the synthesis
            procedure input.
        evaluation_description (str): Description for the evaluation output.

    Returns:
        dspy.Signature: The constructed dspy signature for synthesis evaluation
    """
    signature = {
        "target_material": (
            str,
            dspy.InputField(description=target_material_description),
        ),
        "extracted_recipe": (
            str,
            dspy.InputField(description=extracted_recipe_description),
        ),
        "synthesis_procedure": (
            str,
            dspy.InputField(description=synthesis_procedure_description),
        ),
        "evaluation": (
            SynthesisEvaluation,
            dspy.OutputField(description=evaluation_description),
        ),
    }
    return dspy.make_signature(
        signature_name=signature_name,
        instructions=instructions,
        signature=signature,
    )
