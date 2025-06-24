import torch

from llm_synthesis.models.figure import FigureInfo
from llm_synthesis.models.resnet import (
    FigureClassifier,
)
from llm_synthesis.transformers.figure_extraction.base import (
    FigureExtractorInterface,
)
from llm_synthesis.utils.figure_utils import (
    base64_to_image,
    find_figures_in_markdown,
)


class FigureExtractorMarkdown(FigureExtractorInterface):
    """
    Extracts figures from a markdown text using regex-based markdown parsing.
    """

    def __init__(self):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.classifier = FigureClassifier()

    def forward(self, input: str) -> list[FigureInfo]:
        """
        Extract figures from the given markdown text using markdown parsing.

        Args:
            input (str): The markdown text from which to extract figures.

        Returns:
            list[FigureInfo]: A list of extracted figure information objects.
        """
        figures = find_figures_in_markdown(input)

        for figure in figures:
            pil_image = base64_to_image(figure.base64_data)
            predicted_label = self.classifier.predict(pil_image)
            figure.figure_class = predicted_label

            # Check if the predicted label is a quantitative figure
            if predicted_label in [
                "Bar plots",
                "Contour plot",
                "Graph plots",
                "Scatter plot",
                "Tables",
            ]:
                figure.quantitative = True
            else:
                figure.quantitative = False

        return figures
