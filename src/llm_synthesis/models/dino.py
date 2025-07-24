"""
Segmentation of figures into subplots using DINO."""

import base64
import io

import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor


class FigureSegmenter:
    """
    Segment a figure into subplots using DINO.
    """

    def __init__(self, model_id="IDEA-Research/grounding-dino-base"):
        """
        Initialize the segmenter with the DINO model.

        Args:
            model_id (str): The model ID for the grounding DINO model
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_id
        ).to(self.device)
        self.text_labels = [["a plot"]]  # Labels to detect plots

    def segment(self, image: Image.Image) -> list[Image.Image]:
        """
        Segment a figure into subplots using DINO.

        Args:
            image (Image.Image): PIL Image to segment

        Returns:
            list[Image.Image]: List of cropped subplot images
        """
        # Detect objects using DINO
        detection_results = self._detect_objects(
            image, self.text_labels, box_threshold=0.3, text_threshold=0.3
        )

        # Filter boxes to remove those that cover too much of the image
        filtered_boxes, filtered_scores, filtered_labels = self._filter_boxes(
            detection_results, image.size, max_coverage=0.5
        )

        # If no subplots detected, return the original image
        if len(filtered_boxes) == 0:
            return [image]

        # Expand boxes and crop subplots
        segmented_images = []
        for box in filtered_boxes:
            expanded_box = self._expand_box(box, image.size)
            x_min, y_min, x_max, y_max = expanded_box

            # Crop the subplot from the original image
            crop = image.crop((x_min, y_min, x_max, y_max))
            segmented_images.append(crop)

        return segmented_images

    def _detect_objects(
        self, image, text_labels, box_threshold=0.3, text_threshold=0.3
    ):
        """
        Run grounded object detection on the input image with given labels.
        Returns post-processed results.
        """
        inputs = self.processor(
            images=image, text=text_labels, return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[image.size[::-1]],
        )
        return results[0]

    def _filter_boxes(self, results, image_size, max_coverage=0.9):
        """
        Filter out bounding boxes that cover too much of the image.
        """
        img_width, img_height = image_size
        img_area = img_width * img_height

        filtered_boxes = []
        filtered_scores = []
        filtered_labels = []

        for box, score, label in zip(
            results["boxes"], results["scores"], results["labels"]
        ):
            x_min, y_min, x_max, y_max = box.tolist()
            box_area = (x_max - x_min) * (y_max - y_min)
            coverage = box_area / img_area

            if coverage < max_coverage:
                filtered_boxes.append(box)
                filtered_scores.append(score)
                filtered_labels.append(label)

        return filtered_boxes, filtered_scores, filtered_labels

    def _expand_box(
        self,
        box,
        image_size,
        expand_left_right=0.4,
        expand_bottom=0.3,
        expand_top=0.1,
    ):
        """
        Expand the box by a percentage of its width (left/right symmetrically)
        and bottom side only.
        """
        x_min, y_min, x_max, y_max = box.tolist()
        img_w, img_h = image_size

        width = x_max - x_min
        height = y_max - y_min

        # Expand left/right symmetrically
        x_min_expanded = x_min - (width * expand_left_right / 2)
        x_max_expanded = x_max + (width * expand_left_right / 2)

        # Expand bottom only
        y_max_expanded = y_max + (height * expand_bottom)
        y_min_expanded = y_min - (height * expand_top)

        # Clamp to image bounds
        x_min_expanded = max(0, x_min_expanded)
        x_max_expanded = min(img_w, x_max_expanded)
        y_max_expanded = min(img_h, y_max_expanded)
        y_min_expanded = max(0, y_min_expanded)

        return [x_min_expanded, y_min_expanded, x_max_expanded, y_max_expanded]

    def _image_to_base64(self, image: Image.Image) -> str:
        """
        Convert a PIL Image to a base64-encoded string.

        Args:
            image (Image.Image): PIL Image to convert

        Returns:
            str: Base64-encoded string representation of the image
        """
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
