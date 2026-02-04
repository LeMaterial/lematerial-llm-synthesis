from io import BytesIO
from typing import Literal

import torch
from PIL import Image

from llm_synthesis.models.dino import FigureSegmenter
from llm_synthesis.models.figure import FigureInfo
from llm_synthesis.models.florence import FlorenceSegmenter
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

    def forward(self, input: list[dict[str, bytes | str]]) -> list[FigureInfo]:
        """
        Extract figures from given list of dictionaries containing image data.

        Args:
            input: list[dict[str, bytes | str]]: A list of dictionaries with
            keys 'path' and 'bytes'.

        Returns:
            List[FigureInfo]: A list of FigureInfo objects containing processed
            figure data and metadata.
        """

        all_segmented_images: list[FigureInfo] = []

        print(f"Found {len(input)} figures in the paper.")

        for figure_dict in input:
            figure_path = figure_dict.get("path", "")
            figure_bytes = figure_dict.get("bytes", b"")

            if not isinstance(figure_bytes, bytes):
                print(f"Skipping figure {figure_path}: invalid bytes data")
                continue

            if len(figure_bytes) == 0:
                print(f"Skipping figure {figure_path}: empty bytes data")
                continue

            try:
                # Open and validate the image
                pil_image = Image.open(BytesIO(figure_bytes))

                # Convert to RGB if necessary
                if pil_image.mode != "RGB":
                    pil_image = pil_image.convert("RGB")

                # Load the image data to ensure it's valid
                pil_image.load()

            except Exception as e:
                print(
                    f"Skipping figure {figure_path}: failed to load image - {e}"
                )
                continue

            # Process based on segmenter type
            if self.segmenter_type == "florence":
                all_segmented_images.extend(
                    self._process_with_florence(pil_image, figure_path)
                )
            else:
                all_segmented_images.extend(
                    self._process_with_dino(pil_image, figure_path)
                )

        return all_segmented_images

    def _process_with_florence(
        self, pil_image: Image.Image, figure_path: str
    ) -> list[FigureInfo]:
        """
        Process an image using Florence-2 (detection + classification).

        Args:
            pil_image: PIL Image to process
            figure_path: Path to the original figure for logging

        Returns:
            List of FigureInfo objects for each detected subplot
        """
        results = []

        try:
            detections = self.segmenter.segment_with_labels(pil_image)
            print(f"segm. {len(detections)} subfig. from {figure_path}.")
        except Exception as e:
            print(f"Failed to segment figure {figure_path}: {e}")
            # Fallback: return original image as unknown
            try:
                figure_info = FigureInfo(
                    base64_data=self.segmenter._image_to_base64(pil_image),
                    alt_text=f"Figure from {figure_path}",
                    position=0,
                    context_before="",
                    context_after="",
                    figure_reference=f"{figure_path}_subfigure_1",
                    figure_class="Unknown",
                    quantitative=False,
                )
                return [figure_info]
            except Exception:
                return []

        for i, detection in enumerate(detections):
            try:
                # Florence provides both image and label
                is_quantitative = self.segmenter.is_quantitative(
                    detection.label
                )

                figure_info = FigureInfo(
                    base64_data=self.segmenter._image_to_base64(
                        detection.image
                    ),
                    alt_text=f"Subfigure {i + 1} from {figure_path}",
                    position=0,
                    context_before="",
                    context_after="",
                    figure_reference=f"{figure_path}_subfigure_{i + 1}",
                    figure_class=detection.label,
                    quantitative=is_quantitative,
                )
                results.append(figure_info)

            except Exception as e:
                print(
                    f"Failed to process subfig. {i + 1}",
                    f"from {figure_path}: {e}",
                )
                continue

        return results

    def _process_with_dino(
        self, pil_image: Image.Image, figure_path: str
    ) -> list[FigureInfo]:
        """
        Process an image using DINO segmenter + ResNet classifier (original).

        Args:
            pil_image: PIL Image to process
            figure_path: Path to the original figure for logging

        Returns:
            List of FigureInfo objects for each detected subplot
        """
        results = []

        try:
            segmented_images = self.segmenter.segment(pil_image)
            print(f"segm. {len(segmented_images)} subfig. from {figure_path}.")
        except Exception as e:
            print(f"Failed to segment figure {figure_path}: {e}")
            segmented_images = [pil_image]

        for i, subfigure in enumerate(segmented_images):
            try:
                # Create FigureInfo object for each subfigure
                figure_info = FigureInfo(
                    base64_data=self.segmenter._image_to_base64(subfigure),
                    alt_text=f"Subfigure {i + 1} from {figure_path}",
                    position=0,
                    context_before="",
                    context_after="",
                    figure_reference=f"{figure_path}_subfigure_{i + 1}",
                    figure_class="Unknown",
                    quantitative=False,
                )

                # Classify the subfigure
                try:
                    predicted_label = self.classifier.predict(subfigure)
                    figure_info.figure_class = predicted_label

                    # Check if the predicted label is a quantitative figure
                    if predicted_label in [
                        # "3D objects",
                        # "Algorithm",
                        # "Area chart",
                        "Bar plots",
                        # "Block diagram",
                        "Box plot",
                        "Bubble Chart",
                        "Confusion matrix",
                        "Contour plot",
                        # "Flow chart",
                        # "Geographic map",
                        "Graph plots",
                        "Heat map",
                        "Histogram",
                        # "Mask",
                        # "Medical images",
                        # "Natural images",
                        "Pareto charts",
                        "Pie chart",
                        "Polar plot",
                        "Radar chart",
                        "Scatter plot",
                        # "Sketches",
                        "Surface plot",
                        # "Tables",
                        # "Tree Diagram",
                        "Vector plot",
                        # "Venn Diagram",
                    ]:
                        figure_info.quantitative = True
                    else:
                        figure_info.quantitative = False
                except Exception as e:
                    print(
                        f"Failed to classify subfig. {i + 1}",
                        f"from {figure_path}: {e}",
                    )
                    figure_info.figure_class = "Unknown"
                    figure_info.quantitative = False

                results.append(figure_info)

            except Exception as e:
                print(
                    f"Failed to process subfig. {i + 1}",
                    f"from {figure_path}: {e}",
                )
                continue

        return results
