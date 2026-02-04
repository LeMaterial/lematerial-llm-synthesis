from typing import Literal

import torch

from llm_synthesis.models.dino import FigureSegmenter
from llm_synthesis.models.figure import FigureInfo
from llm_synthesis.models.florence import FlorenceSegmenter
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

    Supports two segmentation backends:
    - "dino": Uses Grounding DINO + ResNet-152 classifier (original)
    - "florence": Uses Florence-2 with LoRA for detection + classification
    """

    def __init__(
        self,
        segmenter: Literal["dino", "florence"] = "dino",
        florence_repo_id: str = (
            "amayuelas/plot-visualization-florence-2-lora-32"
        ),
    ):
        """
        Initialize the figure extractor.

        Args:
            segmenter: Which segmentation backend to use ("dino" or "florence")
            florence_repo_id: HuggingFace repo ID for Florence LoRA adapter
                (only used if segmenter="florence")
        """
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.segmenter_type = segmenter

        if segmenter == "florence":
            self.segmenter = FlorenceSegmenter(repo_id=florence_repo_id)
            self.classifier = None  # Florence handles classification
        else:
            self.segmenter = FigureSegmenter()
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

        all_segmented_images: list[FigureInfo] = []

        print(f"Found {len(figures)} figures in the paper.")

        for figure in figures:
            pil_image = base64_to_image(figure.base64_data)

            if self.segmenter_type == "florence":
                all_segmented_images.extend(
                    self._process_with_florence(pil_image, figure)
                )
            else:
                all_segmented_images.extend(
                    self._process_with_dino(pil_image, figure)
                )

        return all_segmented_images

    def _process_with_florence(
        self, pil_image, figure: FigureInfo
    ) -> list[FigureInfo]:
        """
        Process an image using Florence-2 (detection + classification).

        Args:
            pil_image: PIL Image to process
            figure: Original FigureInfo with context metadata

        Returns:
            List of FigureInfo objects for each detected subplot
        """
        results = []

        try:
            detections = self.segmenter.segment_with_labels(pil_image)
            print(f"Segmented {len(detections)} subfigures (Florence).")
        except Exception as e:
            print(f"Failed to segment figure: {e}")
            # Fallback: return original image
            return [figure]

        for detection in detections:
            is_quantitative = self.segmenter.is_quantitative(detection.label)

            figure_info = FigureInfo(
                base64_data=self.segmenter._image_to_base64(detection.image),
                alt_text=figure.alt_text,
                position=figure.position,
                context_before=figure.context_before,
                context_after=figure.context_after,
                figure_reference=figure.figure_reference,
                figure_class=detection.label,
                quantitative=is_quantitative,
            )
            results.append(figure_info)

        return results

    def _process_with_dino(
        self, pil_image, figure: FigureInfo
    ) -> list[FigureInfo]:
        """
        Process an image using DINO segmenter + ResNet classifier (original).

        Args:
            pil_image: PIL Image to process
            figure: Original FigureInfo with context metadata

        Returns:
            List of FigureInfo objects for each detected subplot
        """
        results = []

        segmented_images = self.segmenter.segment(pil_image)
        print(f"Segmented {len(segmented_images)} subfigures (DINO).")

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

            results.append(figure_info)

        return results
