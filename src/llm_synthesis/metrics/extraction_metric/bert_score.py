"""Script to calculate the BERTScore of two strings of text."""

from bert_score import BERTScorer

from llm_synthesis.metrics.extraction_metric.base import ExtractionMetric


class BERTScore(ExtractionMetric):
    """
    BERTScore extraction metric.
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

        # This is an inefficient implementation.
        # Scorer should be initialized once and
        # reused for multiple calls!
        scorer = BERTScorer()

        return scorer.score(preds, refs)
