from abc import abstractmethod

from llm_synthesis.metrics.base import MetricInterface
from llm_synthesis.models.ontologies import GeneralSynthesisOntology


class TextToTextExtractionMetric(MetricInterface[str]):
    """
    Generic interface for an extraction metric that takes
    two inputs of type string and returns a float.
    """

    @abstractmethod
    def __call__(self, preds: str, refs: str) -> float:
        pass


class TextToOntologyExtractionMetric(
    MetricInterface[GeneralSynthesisOntology]
):
    """
    Generic interface for an extraction metric that takes
    two inputs of type GeneralSynthesisOntology and returns a float.
    """

    @abstractmethod
    def __call__(
        self, preds: GeneralSynthesisOntology, refs: GeneralSynthesisOntology
    ) -> float:
        pass
