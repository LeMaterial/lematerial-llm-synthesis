from io import BytesIO

import torch
from PIL import Image

from llm_synthesis.models.dino import FigureSegmenter
from llm_synthesis.models.resnet import (
    FigureClassifier,
)
from llm_synthesis.transformers.figure_extraction.base import (
    FigureExtractorInterface,
)


class HFFigureExtractor(FigureExtractorInterface):
    """
    Filter images and extract plot data from image bytes
    (as provided from HF dataset).
    """

    def __init__(self):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.classifier = FigureClassifier()
        self.segmenter = FigureSegmenter()

    def forward(
        self, input: list[dict[str, bytes | str]]
    ) -> list[dict[str, bytes | str]]:
        """
        Extract figures from the given markdown text using markdown parsing.

        Args:
            input: list[dict[str, bytes | str]]: A list of dictionaries with
            keys 'path' and 'bytes'.

        Returns:
            List[Dict[str, bytes | str]]: A list of dictionaries with keys
            'path' and 'bytes'.
        """

        all_segmented_images: list[dict[str, bytes | str]] = []

        print(f"Found {len(input)} figures in the paper.")

        for figure_path, figure_bytes in input:
            pil_image = Image.open(BytesIO(figure_bytes))

            segmented_images = self.segmenter.segment(pil_image)

            print(f"Segmented {len(segmented_images)} subfigures.")

            # for subfigure in segmented_images:
            #     # not sure how to fix this because we won't have context
            #     # easily anymore
            #     # down here what we need is to make the classification and
            #     #  extract plot info,
            #     # but then write back to the same format we read in from,
            #     # which is a list of dictionaries that each have two keys
            #     # figure_path and bytes. the figure_paths must be unique
            #     figure_info = FigureInfo(
            #         base64_data=self.segmenter._image_to_base64(subfigure),
            #         alt_text=figure.alt_text,
            #         position=figure.position,
            #         context_before=figure.context_before,
            #         context_after=figure.context_after,
            #         figure_reference=figure.figure_reference,
            #         figure_class=figure.figure_class,
            #         quantitative=figure.quantitative,
            #     )

            # predicted_label = self.classifier.predict(subfigure)
            # figure_info.figure_class = predicted_label

            # # Check if the predicted label is a quantitative figure
            # if predicted_label in [
            #     "Bar plots",
            #     "Line plots",
            #     "Scatter plot",
            # ]:
            #     figure_info.quantitative = True
            # else:
            #     figure_info.quantitative = False

            # all_segmented_images.append(figure_info)

        return all_segmented_images
