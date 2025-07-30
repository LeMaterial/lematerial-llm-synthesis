import base64
import io

from PIL import Image

from llm_synthesis.models.figure import FigureInfo
from llm_synthesis.models.paper import ImageData
from llm_synthesis.models.resnet import FigureClassifier
from llm_synthesis.transformers.figure_extraction.base import (
    ImageFigureExtractorInterface,
)


class ImageFigureExtractor(ImageFigureExtractorInterface):
    """
    Extracts figures directly from image bytes (for HuggingFace datasets).
    """

    def __init__(self):
        self.classifier = FigureClassifier()

    def forward(self, input: list[ImageData]) -> list[FigureInfo]:
        """
        Extract figures from a list of image data.

        Args:
            input: List of ImageData objects containing image bytes and paths

        Returns:
            list[FigureInfo]: A list of extracted figure information objects
        """
        figures = []

        for i, image_data in enumerate(input):
            try:
                # Convert bytes to PIL Image
                image = Image.open(io.BytesIO(image_data.bytes))

                # Convert to base64 for consistency with other figure extractors
                img_buffer = io.BytesIO()
                image.save(img_buffer, format="PNG")
                base64_data = base64.b64encode(img_buffer.getvalue()).decode(
                    "utf-8"
                )

                # Predict figure class
                predicted_label = self.classifier.predict(image)

                # Determine if figure is quantitative
                quantitative = predicted_label in [
                    "Bar plots",
                    # "Contour plot",
                    # "Graph plots",
                    "Scatter plot",
                    "Line plots",
                    # "Tables",
                ]

                # Create FigureInfo object
                figure_info = FigureInfo(
                    base64_data=base64_data,
                    alt_text=f"Figure from {image_data.path}",
                    position=i,  # Use index as position
                    context_before="",  # No context available from image bytes
                    context_after="",  # No context available from image bytes
                    figure_reference=f"Figure {i + 1}",
                    figure_class=predicted_label,
                    quantitative=quantitative,
                )

                figures.append(figure_info)

            except Exception as e:
                print(f"Failed to process image {image_data.path}: {e}")
                continue

        return figures
