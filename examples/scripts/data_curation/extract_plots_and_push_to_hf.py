import logging
from io import BytesIO
from pathlib import Path

import tqdm
from datasets import Dataset, load_dataset
from PIL import Image

from llm_synthesis.transformers.figure_extraction.hf_figure_extractor import (
    HFFigureExtractor,
)
from llm_synthesis.transformers.plot_extraction.claude_extraction.plot_data_extraction import (  # noqa: E501
    ClaudeLinePlotDataExtractor,
)
from llm_synthesis.utils.figure_utils import base64_to_image

# only log info
logging.basicConfig(level=logging.INFO)


def save_image_to_folder(image, save_path: Path, image_name: str):
    """Save an image to the specified folder with a descriptive name."""
    try:
        # Handle FigureInfo objects with base64_data
        if hasattr(image, "base64_data") and image.base64_data:
            try:
                pil_image = base64_to_image(image.base64_data)
                pil_image.save(save_path / f"{image_name}.png")
                return
            except Exception as e:
                logging.error(
                    f"Error converting base64 to image for {image_name}: {e}"
                )
                return

        # Handle PIL Image objects directly
        if hasattr(image, "save") and callable(image.save):
            image.save(save_path / f"{image_name}.png")
        elif hasattr(image, "pil_image") and image.pil_image is not None:
            image.pil_image.save(save_path / f"{image_name}.png")
        else:
            logging.warning(
                f"Could not save image {image_name} - no save method found"
            )
    except Exception as e:
        logging.error(f"Error saving image {image_name}: {e}")


def create_paper_folders(base_path: Path, paper_id: str):
    """Create folder structure for a paper:
    ocr_images, segmented_images, quantitative_images, line_charts."""
    paper_path = base_path / paper_id
    ocr_images_path = paper_path / "ocr_images"
    segmented_path = paper_path / "segmented_images"
    quantitative_path = paper_path / "quantitative_images"
    line_charts_path = paper_path / "line_charts"

    # Create all directories
    ocr_images_path.mkdir(parents=True, exist_ok=True)
    segmented_path.mkdir(parents=True, exist_ok=True)
    quantitative_path.mkdir(parents=True, exist_ok=True)
    line_charts_path.mkdir(parents=True, exist_ok=True)

    return ocr_images_path, segmented_path, quantitative_path, line_charts_path


def main(batch_size: int = 10, config="full", split="chemrxiv"):
    hf_figure_extractor = HFFigureExtractor()

    # Initialize the extractor
    extractor = ClaudeLinePlotDataExtractor(
        model_name="claude-sonnet-4-20250514"
    )

    # Create base directory for saving images
    base_save_path = Path("results/plot_extraction")
    base_save_path.mkdir(parents=True, exist_ok=True)

    total_line_charts = []
    total_quantative_images = []
    total_segmented_images = []
    total_ocr_images = []
    dataset = load_dataset(
        "LeMaterial/LeMat-Synth",
        name=config,
        split=split,
        verification_mode="no_checks",
    )
    df = dataset.to_pandas()
    # Check for duplicates
    duplicates = df[df.duplicated(subset=["paper_doi"], keep=False)]
    logging.info(f"Found {len(duplicates)} duplicate papers")
    logging.info(duplicates["paper_doi"].value_counts().head())

    # remove duplicates
    df = df.drop_duplicates(subset=["paper_doi"])

    processed_count = 0  # Add a separate counter

    for idx, row in tqdm.tqdm(df.iterrows(), total=len(df)):
        paper_id = row.get("paper_doi", f"paper_{idx}")
        logging.info(f"Processing paper: {paper_id}")

        if row["images"] is None:
            logging.info("Skipping paper with no images")
            continue

        # Create folders for this paper
        ocr_images_path, segmented_path, quantitative_path, line_charts_path = (
            create_paper_folders(base_save_path, paper_id)
        )

        ocr_images = row["images"]
        for i, img in enumerate(ocr_images):
            # Handle raw image data from the dataset
            if isinstance(img, dict) and "bytes" in img and img["bytes"]:
                try:
                    # Convert bytes to PIL Image and save
                    pil_image = Image.open(BytesIO(img["bytes"]))
                    pil_image.save(ocr_images_path / f"ocr_image_{i + 1}.png")
                except Exception as e:
                    logging.error(f"Error saving OCR image {i + 1}: {e}")
            else:
                # Try the general save method as fallback
                save_image_to_folder(img, ocr_images_path, f"ocr_image_{i + 1}")

        segmented_images = hf_figure_extractor.forward(row["images"])
        logging.info(f"Found {len(segmented_images)} figures in the paper.")

        # Save segmented images
        for i, img in enumerate(segmented_images):
            # Include the figure class in the filename for better identification
            figure_class = getattr(img, "figure_class", "unknown")
            # Clean the figure class name for use in filename
            # (remove spaces, special chars)
            clean_class = (
                figure_class.replace(" ", "_")
                .replace("/", "_")
                .replace("\\", "_")
            )
            filename = f"segmented_figure_{i + 1}_{clean_class}"
            save_image_to_folder(img, segmented_path, filename)

        quantative_images = [
            img for img in segmented_images if img.quantitative
        ]
        logging.info(
            f"Found {len(quantative_images)} quantitative figures in the paper."
        )

        # Save quantitative images
        for i, img in enumerate(quantative_images):
            # Include the figure class in the filename for better identification
            figure_class = getattr(img, "figure_class", "unknown")
            # Clean the figure class name for use in filename
            # (remove spaces, special chars)
            clean_class = (
                figure_class.replace(" ", "_")
                .replace("/", "_")
                .replace("\\", "_")
            )
            filename = f"quantitative_figure_{i + 1}_{clean_class}"
            save_image_to_folder(img, quantitative_path, filename)

        quantitative_images_classes_for_extraction = [
            "Bar plots",
            "Contour plot",
            "Graph plots",
            "Scatter plot",
            "Surface plot",
            "Vector plot",
        ]

        line_charts = [
            img
            for img in quantative_images
            if img.figure_class in quantitative_images_classes_for_extraction
        ]
        logging.info(f"Found {len(line_charts)} line charts in paper.")

        # Save line charts
        for i, img in enumerate(line_charts):
            image_name = f"line_chart_{i + 1}_{img.figure_class}"
            save_image_to_folder(img, line_charts_path, image_name)

        total_line_charts.extend(line_charts)
        total_quantative_images.extend(quantative_images)
        total_segmented_images.extend(segmented_images)
        total_ocr_images.extend(ocr_images)

        plot_data = []

        for line_chart in line_charts:
            extracted_data = extractor.forward(line_chart)
            # extracted_data.figure_class = line_chart.figure_class
            logging.info(extracted_data)
            plot_data.append(extracted_data)
            logging.info("-" * 100)

        row["plot_data"] = plot_data
        processed_count += 1  # Increment the counter

        if processed_count % batch_size == 0:
            logging.info(
                f"Pushing batch {processed_count // batch_size} to hub",
                f"(processed {processed_count} samples)",
            )
            ds = Dataset.from_pandas(df)
            ds.push_to_hub(
                "LeMaterial/LeMat-Synth",
                config_name=config,
                split=split,
                create_pr=True,
            )
            df = dataset.to_pandas()

    # Push any remaining data
    if processed_count % batch_size != 0:
        logging.info(
            "Pushing final batch to hub",
            f"(processed {processed_count} samples total)",
        )
        ds = Dataset.from_pandas(df)
        ds.push_to_hub(
            "LeMaterial/LeMat-Synth",
            config_name=config,
            split=split,
            create_pr=True,
        )

    logging.info(f"Total segmented images: {len(total_segmented_images)}")
    logging.info(f"Total ocr images: {len(total_ocr_images)}")
    logging.info(f"Total line charts: {len(total_line_charts)}")
    logging.info(f"Total quantative images: {len(total_quantative_images)}")


if __name__ == "__main__":
    main(batch_size=100)
