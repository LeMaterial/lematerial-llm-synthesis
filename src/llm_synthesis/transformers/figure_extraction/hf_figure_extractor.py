from io import BytesIO

import torch
from PIL import Image

from llm_synthesis.models.dino import FigureSegmenter
from llm_synthesis.models.figure import FigureInfo
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
    ) -> list[FigureInfo]:
        """
        Extract figures from the given list of dictionaries containing image data.

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
            figure_path = figure_dict.get('path', '')
            figure_bytes = figure_dict.get('bytes', b'')
            
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
                if pil_image.mode != 'RGB':
                    pil_image = pil_image.convert('RGB')
                
                # Verify the image is valid
                pil_image.verify()
                
            except Exception as e:
                print(f"Skipping figure {figure_path}: failed to load image - {e}")
                continue

            try:
                segmented_images = self.segmenter.segment(pil_image)
                print(f"Segmented {len(segmented_images)} subfigures from {figure_path}.")
            except Exception as e:
                print(f"Failed to segment figure {figure_path}: {e}")
                # If segmentation fails, use the original image
                segmented_images = [pil_image]

            for i, subfigure in enumerate(segmented_images):
                try:
                    # Create FigureInfo object for each subfigure
                    figure_info = FigureInfo(
                        base64_data=self.segmenter._image_to_base64(subfigure),
                        alt_text=f"Subfigure {i+1} from {figure_path}",
                        position=0,  # No position context available from HF dataset
                        context_before="",  # No context available from HF dataset
                        context_after="",   # No context available from HF dataset
                        figure_reference=f"{figure_path}_subfigure_{i+1}",
                        figure_class="Unknown",  # Will be updated by classifier
                        quantitative=False,  # Will be updated based on classification
                    )

                    # Classify the subfigure
                    try:
                        predicted_label = self.classifier.predict(subfigure)
                        figure_info.figure_class = predicted_label

                        # Check if the predicted label is a quantitative figure
                        if predicted_label in [
                            "Bar plots",
                            "Line plots",
                            "Scatter plot",
                        ]:
                            figure_info.quantitative = True
                        else:
                            figure_info.quantitative = False
                    except Exception as e:
                        print(f"Failed to classify subfigure {i+1} from {figure_path}: {e}")
                        # Keep the default values if classification fails
                        figure_info.figure_class = "Unknown"
                        figure_info.quantitative = False

                    all_segmented_images.append(figure_info)
                    
                except Exception as e:
                    print(f"Failed to process subfigure {i+1} from {figure_path}: {e}")
                    continue

        return all_segmented_images
