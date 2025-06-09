import random

from llm_synthesis.metrics.extraction_metric.base import ExtractionMetric


class DummyExtractionMetric(ExtractionMetric):
    """
    Dummy extraction metric that returns a random float.
    """

    def __call__(self, preds: str, refs: str) -> float:
        """
        Dummy extraction metric that returns a random float.

        Args:
            preds: The predicted string.
            refs: The reference string.

        Returns:
            A random float.
        """
        return random.random()
