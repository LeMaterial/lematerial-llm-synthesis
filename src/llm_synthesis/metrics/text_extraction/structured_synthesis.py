import logging

from llm_synthesis.metrics.text_extraction.base import (
    TextToOntologyExtractionMetric,
)
from llm_synthesis.models.ontologies import GeneralSynthesisOntology

logger = logging.getLogger(__name__)


def _log_steps(label: str, ontology: GeneralSynthesisOntology) -> None:
    """Dump an ontology's steps at DEBUG level (verbose; off by default)."""
    if not logger.isEnabledFor(logging.DEBUG):
        return
    logger.debug("%s.steps: %s", label, ontology.steps)
    logger.debug("len(%s.steps): %d", label, len(ontology.steps))
    for step in ontology.steps:
        logger.debug("--------------------------------")
        logger.debug("%r", step)
        logger.debug("step_number=%s", step.step_number)
        logger.debug("action=%s", step.action)
        logger.debug("description=%s", step.description)
        logger.debug("materials=%s", step.materials)
        logger.debug("equipment=%s", step.equipment)


class NumberCheckerMetric(TextToOntologyExtractionMetric):
    """
    Metric for checking if the number of steps in the synthesis is correct.
    """

    def __call__(
        self, preds: GeneralSynthesisOntology, refs: GeneralSynthesisOntology
    ) -> float:
        _log_steps("preds", preds)
        _log_steps("refs", refs)
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
