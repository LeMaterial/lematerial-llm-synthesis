from torchmetrics.text import CharErrorRate, WordErrorRate

from llm_synthesis.metrics.extraction_metric.base import (
    FigureToQuantiativeDataExtractionMetric,
)


class CER(FigureToQuantiativeDataExtractionMetric):
    """
    Metric for create the character error rate of the OCR.
    """

    def __call__(self, preds: str, refs: str) -> float:
        """
        Calculate the character error rate of the OCR.
        """
        cer = CharErrorRate()
        return cer(preds, refs)


class WER(FigureToQuantiativeDataExtractionMetric):
    """
    Metric for create the word error rate of the OCR.
    """

    def __call__(self, preds: str, refs: str) -> float:
        """
        Calculate the word error rate of the OCR.
        """
        wer = WordErrorRate()
        return wer(preds, refs)
