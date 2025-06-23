from llm_synthesis.models.figure import FigureInfo
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
        self.image_classifier = (
            "foo"  # TODO: add image classifier, loading function or similar
        )

    def forward(self, input: str) -> list[FigureInfo]:
        """
        Extract figures from the given markdown text using markdown parsing.

        Args:
            input (str): The markdown text from which to extract figures.

        Returns:
            list[FigureInfo]: A list of extracted figure information objects.
        """
        base64_images = find_figures_in_markdown(input)

        pil_images = [
            base64_to_image(base64_data) for base64_data in base64_images
        ]

        for image in pil_images:
            pass
