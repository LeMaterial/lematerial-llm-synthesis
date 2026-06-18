"""
Segmentation of figures into subplots using Florence-2 with LoRA adapters.

This module provides an alternative to the DINO-based segmenter, using
Florence-2 fine-tuned with LoRA adapters for plot detection. The model
directly outputs both bounding boxes AND classification labels
(quantitative/qualitative), eliminating the need for a separate classifier.
"""

import base64
import io
import logging
import re
from dataclasses import dataclass

import torch
from peft import PeftModel
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """A single detection result with bounding box and label."""

    bbox: list[float]  # [x1, y1, x2, y2] in pixel coordinates
    label: str  # "quantitative plot" or "qualitative plot"
    image: Image.Image  # Cropped subplot image


class FlorenceSegmenter:
    """
    Segment a figure into subplots using Florence-2 with LoRA adapters.

    This segmenter uses a fine-tuned Florence-2 model that detects subplots
    and classifies them as quantitative or qualitative in a single pass.
    """

    def __init__(
        self,
        repo_id: str = "amayuelas/plot-visualization-florence-2-lora-32",
        base_model: str = "microsoft/Florence-2-base-ft",
        device: str | None = None,
    ):
        """
        Initialize the segmenter with Florence-2 + LoRA model.

        Args:
            repo_id: HuggingFace repo ID for the LoRA adapter
            base_model: Base Florence-2 model identifier
            device: to run model on ('cuda', 'cpu', 'mps', or None for auto)
        """
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = device
        self.repo_id = repo_id
        self.base_model = base_model

        # Store last detections for classification lookup
        self._last_detections: list[Detection] = []

        self._load_model()

    def _load_model(self):
        """Load the Florence-2 base model with LoRA adapters."""
        logger.info("Loading Florence-2 base model: %s", self.base_model)
        self.processor = AutoProcessor.from_pretrained(
            self.base_model, trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            trust_remote_code=True,
            torch_dtype=torch.float16
            if self.device != "cpu"
            else torch.float32,
            attn_implementation="eager",
        )

        logger.info("Loading LoRA adapters from: %s", self.repo_id)
        model = PeftModel.from_pretrained(model, self.repo_id)

        logger.info("Merging LoRA adapters with base model...")
        model = model.merge_and_unload()

        self.model = model.to(self.device)

        # Newer transformers removed forced_bos_token_id from
        # Florence2LanguageConfig; add it so generate() does not fail.
        lang_cfg = getattr(
            getattr(self.model, "language_model", None), "config", None
        )
        if lang_cfg is not None and not hasattr(
            lang_cfg, "forced_bos_token_id"
        ):
            lang_cfg.forced_bos_token_id = None

        logger.info("Florence-2 model loaded on %s", self.device)

    def _parse_output(self, output_text: str) -> list[dict]:
        """
        Parse Florence-2 object detection output.

        Expected format: <loc_x1><loc_y1><loc_x2><loc_y2>label<loc_x1>...

        Returns:
            List of dicts with 'bbox' (normalized 0-1000) and 'label' keys
        """
        pattern = r"<loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)>([^<]+?)(?=<loc_|</s>|$)"  # noqa: E501
        matches = re.findall(pattern, output_text)

        detections = []
        for match in matches:
            x1, y1, x2, y2, label = match
            detections.append(
                {
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "label": label.strip(),
                }
            )

        return detections

    def _denormalize_bbox(
        self, bbox: list[int], width: int, height: int
    ) -> list[float]:
        """Convert normalized (0-1000) bbox to pixel coordinates."""
        x1, y1, x2, y2 = bbox
        return [
            (x1 / 1000) * width,
            (y1 / 1000) * height,
            (x2 / 1000) * width,
            (y2 / 1000) * height,
        ]

    def _expand_box(
        self,
        bbox: list[float],
        image_size: tuple[int, int],
        expand_left_right: float = 0.05,
        expand_top_bottom: float = 0.05,
    ) -> list[float]:
        """
        Expand the bounding box by a percentage to capture full plot.

        Args:
            bbox: [x1, y1, x2, y2] in pixel coordinates
            image_size: (width, height) of the image
            expand_left_right: Fraction to expand horizontally
            expand_top_bottom: Fraction to expand vertically

        Returns:
            Expanded bbox clamped to image bounds
        """
        x1, y1, x2, y2 = bbox
        img_w, img_h = image_size

        width = x2 - x1
        height = y2 - y1

        # Expand symmetrically
        x1_expanded = x1 - (width * expand_left_right)
        x2_expanded = x2 + (width * expand_left_right)
        y1_expanded = y1 - (height * expand_top_bottom)
        y2_expanded = y2 + (height * expand_top_bottom)

        # Clamp to image bounds
        x1_expanded = max(0, x1_expanded)
        x2_expanded = min(img_w, x2_expanded)
        y1_expanded = max(0, y1_expanded)
        y2_expanded = min(img_h, y2_expanded)

        return [x1_expanded, y1_expanded, x2_expanded, y2_expanded]

    def segment(self, image: Image.Image) -> list[Image.Image]:
        """
        Segment a figure into subplots using Florence-2.

        This method is compatible with the FigureSegmenter interface.
        After calling this method, use get_last_detections() to access
        classification labels.

        Args:
            image: PIL Image to segment

        Returns:
            List of cropped subplot images
        """
        detections = self.segment_with_labels(image)
        return [det.image for det in detections]

    def segment_with_labels(self, image: Image.Image) -> list[Detection]:
        """
        Segment a figure and return detections with labels.

        This method returns full detection information including
        bounding boxes, labels, and cropped images.

        Args:
            image: PIL Image to segment

        Returns:
            List of Detection objects with bbox, label, and image
        """
        # Ensure RGB
        if image.mode != "RGB":
            image = image.convert("RGB")

        width, height = image.size

        # Prepare inputs
        prompt = "<OD>"
        inputs = self.processor(
            text=prompt, images=image, return_tensors="pt"
        ).to(self.device)

        if self.device != "cpu":
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)

        # Run inference
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3,
                use_cache=False,
            )

        generated_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=False
        )[0]

        # Parse detections
        raw_detections = self._parse_output(generated_text)

        # Filter out detections that cover too much of the image
        filtered_detections = []
        img_area = width * height

        for det in raw_detections:
            bbox = self._denormalize_bbox(det["bbox"], width, height)
            x1, y1, x2, y2 = bbox
            box_area = (x2 - x1) * (y2 - y1)
            coverage = box_area / img_area

            # Skip if covers more than 90% of image
            if coverage < 0.9:
                filtered_detections.append(
                    {"bbox": bbox, "label": det["label"]}
                )

        # If no subplots detected, return the original image as unknown
        if len(filtered_detections) == 0:
            detection = Detection(
                bbox=[0, 0, width, height],
                label="unknown",
                image=image,
            )
            self._last_detections = [detection]
            return [detection]

        # Expand boxes and crop subplots
        detections = []
        for det in filtered_detections:
            expanded_bbox = self._expand_box(det["bbox"], (width, height))
            x1, y1, x2, y2 = expanded_bbox

            # Crop the subplot
            crop = image.crop((int(x1), int(y1), int(x2), int(y2)))

            detection = Detection(
                bbox=expanded_bbox,
                label=det["label"],
                image=crop,
            )
            detections.append(detection)

        self._last_detections = detections
        return detections

    def get_last_detections(self) -> list[Detection]:
        """
        Get the detections from the last segment() call.

        Returns:
            List of Detection objects from the last segmentation
        """
        return self._last_detections

    def is_quantitative(self, label: str) -> bool:
        """
        Check if a label indicates a quantitative plot.

        Args:
            label: The detection label from Florence-2

        Returns:
            True if the label indicates a quantitative plot
        """
        return "quantitative" in label.lower()

    def _image_to_base64(self, image: Image.Image) -> str:
        """
        Convert a PIL Image to a base64-encoded string.

        Args:
            image: PIL Image to convert

        Returns:
            Base64-encoded string representation of the image
        """
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
