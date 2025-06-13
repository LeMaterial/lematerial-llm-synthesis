from abc import abstractmethod

from llm_synthesis.metrics.base import MetricInterface


class TextToTextExtractionMetric(MetricInterface[str]):
    """
    Generic interface for an extraction metric that takes
    two inputs of type string and returns a float.
    """

    @abstractmethod
    def __call__(self, preds: str, refs: str) -> float:
        pass
