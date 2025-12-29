from abc import abstractmethod

from llm_synthesis.metrics.base import MetricInterface
from llm_synthesis.models.plot import ExtractedLinePlotData


class ExtractionMetric(MetricInterface[str]):
    """
    Generic interface for an extraction metric that takes
    two inputs of type string and returns a float.
    """

    @abstractmethod
    def __call__(self, preds: str, refs: str) -> float:
        pass


class LinePlotExtractionMetric(MetricInterface[ExtractedLinePlotData]):
    """
    Generic interface for a line plot extraction metric that takes
    two inputs of type ExtractedLinePlotData and returns a float.
    """

    @abstractmethod
    def __call__(
        self, preds: ExtractedLinePlotData, refs: ExtractedLinePlotData
    ) -> float:
        pass
