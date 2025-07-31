from llm_synthesis.metrics.extraction.base import (
    TextToOntologyExtractionMetric,
)
from llm_synthesis.models.ontologies import GeneralSynthesisOntology


class NumberCheckerMetric(TextToOntologyExtractionMetric):
    """
    Metric for checking if the number of steps in the synthesis is correct.
    """

    def __call__(
        self, preds: GeneralSynthesisOntology, refs: GeneralSynthesisOntology
    ) -> float:
        print(f"preds.steps: {preds.steps}")
        print(f"len(preds.steps): {len(preds.steps)}")
        for step in preds.steps:
            print("--------------------------------")
            print(step)
            print(step.step_number)
            print(step.action)
            print(step.description)
            print(step.materials)
            print(step.equipment)

        print("--------------------------------")
        print(f"refs.steps: {refs.steps}")
        print(f"len(refs.steps): {len(refs.steps)}")
        for step in refs.steps:
            print("--------------------------------")
            print(step)
            print(step.step_number)
            print(step.action)
            print(step.description)
            print(step.materials)
            print(step.equipment)

        if len(preds.steps) != len(refs.steps):
            return 0
        return 1


class MaterialsCheckerMetric(TextToOntologyExtractionMetric):
    """
    Metric for checking if the materials in the synthesis are correct.
    """

    def __call__(
        self, preds: GeneralSynthesisOntology, refs: GeneralSynthesisOntology
    ) -> float:
        if set(preds.starting_materials) != set(refs.starting_materials):
            return 0
        return 1


class TargetCheckerMetric(TextToOntologyExtractionMetric):
    """
    Metric for checking if the target in the synthesis is correct.
    """

    def __call__(
        self, preds: GeneralSynthesisOntology, refs: GeneralSynthesisOntology
    ) -> float:
        if preds.target_compound != refs.target_compound:
            return 0
        return 1
