import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin, hf_hub_download
from PIL import Image
from pydantic import BaseModel, Field
from torchvision import models, transforms


class ModelConfig(BaseModel):
    """Configuration for the ResNet model."""

    num_classes: int = 28
    repo_id: str = "sehaba95/ResNet-152-DocFigure"
    filename: str = "pytorch_model.bin"


class TransformConfig(BaseModel):
    """Image preprocessing transformations."""

    size: tuple = (224, 224)
    mean: list[float] = [0.485, 0.456, 0.406]
    std: list[float] = [0.229, 0.224, 0.225]


class LabelConfig(BaseModel):
    """Label mapping for prediction output."""

    labels: list[str] = Field(
        default=[
            "3D objects",
            "Algorithm",
            "Area chart",
            "Bar plots",
            "Block diagram",
            "Box plot",
            "Bubble Chart",
            "Confusion matrix",
            "Contour plot",
            "Flow chart",
            "Geographic map",
            "Graph plots",
            "Heat map",
            "Histogram",
            "Mask",
            "Medical images",
            "Natural images",
            "Pareto charts",
            "Pie chart",
            "Polar plot",
            "Radar chart",
            "Scatter plot",
            "Sketches",
            "Surface plot",
            "Tables",
            "Tree Diagram",
            "Vector plot",
            "Venn Diagram",
        ]
    )


class ResNetDocfigModel(nn.Module, PyTorchModelHubMixin):
    """Custom ResNet-152 model for DocFigure classification."""

    def __init__(self, config: ModelConfig = ModelConfig()):
        super().__init__()
        self.config = config

        # Load base ResNet-152 without pretrained weights
        self.resnet = models.resnet152(weights=None)

        # Replace final fully connected layer
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, self.config.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.resnet(x)


class FigureClassifier:
    def __init__(
        self,
        model_config: ModelConfig = ModelConfig(),
        label_config: LabelConfig = LabelConfig(),
        transform_config: TransformConfig = TransformConfig(),
    ):
        """
        Initializes the figure classifier with model, transforms, and device.

        Args:
            model_config: Configuration for the model.
            label_config: Configuration for output labels.
            transform_config: Configuration for image preprocessing.
        """
        self.model_config = model_config
        self.label_config = label_config
        self.transform_config = transform_config
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Initialize transform
        self.transform = self._get_transform()

        # Load model
        self.model = self._load_model().to(self.device).eval()

    def _get_transform(self) -> transforms.Compose:
        """Builds image preprocessing pipeline from config."""
        return transforms.Compose(
            [
                transforms.Resize(
                    self.transform_config.size,
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=self.transform_config.mean,
                    std=self.transform_config.std,
                ),
            ]
        )

    def _load_model(self) -> ResNetDocfigModel:
        """Loads the trained model from Hugging Face Hub."""
        try:
            model_path = hf_hub_download(
                repo_id=self.model_config.repo_id,
                filename=self.model_config.filename,
            )
            state_dict = torch.load(model_path, weights_only=False)

            model = ResNetDocfigModel(self.model_config)
            model.load_state_dict(state_dict)

            return model

        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}") from e

    def predict(self, image_input: Image.Image) -> str:
        """
        Predicts the label of an input image.

        Args:
            image_input: A PIL Image object in RGB mode.

        Returns:
            Predicted label name.
        """
        if not isinstance(image_input, Image.Image):
            raise TypeError(
                "Expected image_input to be a PIL Image.",
                f"Got {type(image_input)}.",
            )

        try:
            image = image_input.convert("RGB")
        except Exception as e:
            raise ValueError(
                "Invalid PIL Image object: cannot convert to RGB mode."
            ) from e

        image_tensor = self.transform(image).to(self.device)  # type: ignore
        image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension

        with torch.no_grad():
            outputs = self.model(image_tensor)

        _, preds = torch.max(outputs, 1)
        return self.label_config.labels[preds.item()]  # type: ignore
