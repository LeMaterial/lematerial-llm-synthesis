"""Inference script for Florence-2 with PEFT/LoRA adapters from HuggingFace Hub."""

import argparse
import re
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import torch
from peft import PeftModel
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor


def parse_florence_output(output_text: str) -> list[dict]:
    """Parse Florence-2 object detection output.

    Expected format: <loc_x1><loc_y1><loc_x2><loc_y2>label<loc_x1>...

    Returns:
        List of dicts with 'bbox' and 'label' keys
    """
    pattern = (
        r"<loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)>([^<]+?)(?=<loc_|</s>|$)"
    )
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


def denormalize_bbox(bbox: list[int], width: int, height: int) -> list[float]:
    """Convert normalized (0-1000) bbox to pixel coordinates."""
    x1, y1, x2, y2 = bbox
    return [
        (x1 / 1000) * width,
        (y1 / 1000) * height,
        (x2 / 1000) * width,
        (y2 / 1000) * height,
    ]


def visualize_detections(
    image_path: Path,
    detections: list[dict],
    save_path: Path | None = None,
    show: bool = True,
):
    """Visualize detected subplots on the image."""
    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    _, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image)

    colors = {
        "quantitative plot": "red",
        "qualitative plot": "blue",
    }

    for det in detections:
        bbox = denormalize_bbox(det["bbox"], width, height)
        x1, y1, x2, y2 = bbox
        label = det["label"]
        color = colors.get(label, "green")

        rect = patches.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            linewidth=2,
            edgecolor=color,
            facecolor="none",
        )
        ax.add_patch(rect)

        ax.text(
            x1,
            y1 - 5,
            label,
            color=color,
            fontsize=10,
            weight="bold",
            bbox={"facecolor": "white", "alpha": 0.7, "pad": 2},
        )

    ax.axis("off")
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"Visualization saved to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def load_model_from_hub(
    repo_id: str,
    base_model: str = "microsoft/Florence-2-base-ft",
    device: str = "cuda",
):
    """Load Florence-2 base model with PEFT/LoRA adapters from HuggingFace Hub.

    Args:
        repo_id: HuggingFace repo ID (e.g., "username/model-name")
        base_model: Base model identifier
        device: Device to load model on

    Returns:
        Tuple of (model, processor)
    """
    print(f"Loading base model: {base_model}")
    processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        attn_implementation="eager",
    )

    print(f"Loading LoRA adapters from HuggingFace Hub: {repo_id}")
    model = PeftModel.from_pretrained(model, repo_id)

    print("Merging LoRA adapters with base model...")
    model = model.merge_and_unload()

    model = model.to(device)

    return model, processor


def predict_image(
    repo_id: str,
    image_path: Path,
    base_model: str = "microsoft/Florence-2-base-ft",
    device: str = "cuda",
    visualize: bool = True,
    save_path: Path | None = None,
) -> list[dict]:
    """Run inference on a single image using model from HuggingFace Hub.

    Args:
        repo_id: HuggingFace repo ID for the LoRA adapter
        image_path: Path to the input image
        base_model: Base model identifier
        device: Device to run inference on ('cuda', 'cpu', or 'mps')
        visualize: Whether to visualize the results
        save_path: Optional path to save visualization

    Returns:
        List of detection dicts
    """
    model, processor = load_model_from_hub(repo_id, base_model, device)

    print(f"\nProcessing image: {image_path}")
    image = Image.open(image_path).convert("RGB")

    prompt = "<OD>"
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
    inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)

    print("Running inference...")
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3,
            use_cache=False,
        )

    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    print(f"\nRaw output: {generated_text}\n")

    detections = parse_florence_output(generated_text)

    print(f"Found {len(detections)} subplots:")
    for i, det in enumerate(detections, 1):
        print(f"  {i}. {det['label']} - bbox: {det['bbox']}")

    if visualize:
        visualize_detections(image_path, detections, save_path)

    return detections


def predict_directory(
    repo_id: str,
    image_dir: Path,
    output_dir: Path | None = None,
    base_model: str = "microsoft/Florence-2-base-ft",
    device: str = "cuda",
    limit: int | None = None,
) -> dict[str, list[dict]]:
    """Run inference on all images in a directory.

    Args:
        repo_id: HuggingFace repo ID for the LoRA adapter
        image_dir: Directory containing images
        output_dir: Optional directory to save visualizations
        base_model: Base model identifier
        device: Device to run inference on
        limit: Optional limit on number of images to process

    Returns:
        Dict mapping image names to detection lists
    """
    image_files = sorted(list(Path(image_dir).glob("*.jpg"))) + sorted(
        list(Path(image_dir).glob("*.png"))
    )

    if limit:
        image_files = image_files[:limit]

    print(f"Processing {len(image_files)} images from {image_dir}")

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    model, processor = load_model_from_hub(repo_id, base_model, device)

    results = {}

    for image_path in image_files:
        print(f"\n{'=' * 80}")
        print(f"Processing: {image_path.name}")
        print(f"{'=' * 80}")

        image = Image.open(image_path).convert("RGB")
        prompt = "<OD>"
        inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)

        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3,
                use_cache=False,
            )

        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        detections = parse_florence_output(generated_text)

        print(f"Found {len(detections)} subplots")
        for i, det in enumerate(detections, 1):
            print(f"  {i}. {det['label']}")

        results[image_path.name] = detections

        if output_dir:
            save_path = output_dir / f"{image_path.stem}_detected.png"
            visualize_detections(image_path, detections, save_path=save_path, show=False)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run inference using model from HuggingFace Hub"
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="amayuelas/plot-visualization-florence-2-lora-32",
        help="HuggingFace repo ID for the LoRA adapter",
    )
    parser.add_argument(
        "--image",
        type=Path,
        help="Path to a single image for inference",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        help="Directory containing images for batch inference",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs_hf/predictions"),
        help="Directory to save visualizations",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu", "mps"],
        help="Device to run inference on",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of images to process (for batch mode)",
    )
    parser.add_argument(
        "--no-visualize",
        action="store_true",
        help="Disable visualization",
    )

    args = parser.parse_args()

    if args.image:
        if not args.image.exists():
            print(f"Error: Image not found at {args.image}")
            return

        save_path = args.output_dir / f"{args.image.stem}_detected.png"
        predict_image(
            repo_id=args.repo_id,
            image_path=args.image,
            device=args.device,
            visualize=not args.no_visualize,
            save_path=save_path,
        )

    elif args.image_dir:
        if not args.image_dir.exists():
            print(f"Error: Directory not found at {args.image_dir}")
            return

        predict_directory(
            repo_id=args.repo_id,
            image_dir=args.image_dir,
            output_dir=args.output_dir,
            device=args.device,
            limit=args.limit,
        )

    else:
        # Default: run on example image
        image_path = Path("deart/test/image_0003.jpg")
        if not image_path.exists():
            print(f"Error: Example image not found at {image_path}")
            print("Please provide --image or --image-dir argument")
            return

        predict_image(
            repo_id=args.repo_id,
            image_path=image_path,
            device=args.device,
            visualize=not args.no_visualize,
            save_path=args.output_dir / "image_0003_detected.png",
        )


if __name__ == "__main__":
    main()