from llm_synthesis.metrics.extraction_metric.base import (
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
        if set(preds.materials) != set(refs.materials):
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
