import torch

from llm_synthesis.models.dino import FigureSegmenter
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
        self.segmenter = FigureSegmenter()

    def forward(self, input: str) -> list[FigureInfo]:
        """
        Extract figures from the given markdown text using markdown parsing.

        Args:
            input (str): The markdown text from which to extract figures.

        Returns:
            list[FigureInfo]: A list of extracted figure information objects.
        """
        figures = find_figures_in_markdown(input)

        all_segmented_images: list[FigureInfo] = []

        print(f"Found {len(figures)} figures in the paper.")

        for figure in figures:
            pil_image = base64_to_image(figure.base64_data)

            segmented_images = self.segmenter.segment(pil_image)

            print(f"Segmented {len(segmented_images)} subfigures.")

            for subfigure in segmented_images:
                figure_info = FigureInfo(
                    base64_data=self.segmenter._image_to_base64(subfigure),
                    alt_text=figure.alt_text,
                    position=figure.position,
                    context_before=figure.context_before,
                    context_after=figure.context_after,
                    figure_reference=figure.figure_reference,
                    figure_class=figure.figure_class,
                    quantitative=figure.quantitative,
                )

                predicted_label = self.classifier.predict(subfigure)
                figure_info.figure_class = predicted_label

                # Check if the predicted label is a quantitative figure
                if predicted_label in [
                    "Line plots",
                ]:
                    figure_info.quantitative = True
                else:
                    figure_info.quantitative = False

                all_segmented_images.append(figure_info)

        return all_segmented_images
