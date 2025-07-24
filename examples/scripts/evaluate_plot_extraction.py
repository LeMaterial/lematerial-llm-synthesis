import base64
import json
import logging
import os
from io import BytesIO

import hydra
import pandas as pd
from hydra.utils import instantiate
from omegaconf import DictConfig
from PIL import Image
from tqdm import tqdm
from pathlib import Path

from llm_synthesis.models.figure import FigureInfoWithPaper
from llm_synthesis.models.plot import DataPoint, DataSeries, ExtractedPlotData, PlotMetadata
from llm_synthesis.transformers.plot_extraction.base import PlotDataExtractorInterface
from llm_synthesis.utils.figure_utils import find_figures_in_markdown

from llm_synthesis.transformers.plot_extraction.claude_extraction.plot_data_extraction import (
    ClaudeLinePlotDataExtractor,
)
from llm_synthesis.models.plot import ExtractedLinePlotData
    
# Get default base dir
base_dir = Path(__file__).parent.parent.parent

def load_ground_truth(json_path: str) -> list[ExtractedPlotData]:
    """Loads a ground truth JSON into an list ofExtractedPlotData object."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # v0, Extract data from the first subplot
    subplot = data["subplots"][0]
    coordinates = subplot["coordinates"]
    x_label = subplot["x_label"]
    y_label = subplot["y_label"]
    
    # Create metadata
    metadata = PlotMetadata(
        x_axis_label=x_label,
        left_y_axis_label=y_label,
        is_dual_axis=False
    )
    
    # Create data series
    data_series = []
    for series_name, points in coordinates.items():
        # Create data points for this series
        data_points = []
        for point in points:
            x, y = point
            data_points.append(DataPoint(
                x=x,
                y=y,
                series_name=series_name,
                axis="left"
            ))
        
        # Create the data series
        series = DataSeries(
            name=series_name,
            points=data_points,
            axis="left"
        )
        data_series.append(series)
    
    if not data_series:
        raise ValueError(f"No data found in {json_path}")
    
    # Create and return list of ExtractedPlotData
    return [ExtractedPlotData(
        metadata=metadata,
        data_series=data_series,
        technical_takeaways=[]  # Empty for ground truth
    )]

def image_to_base64_data_uri(image_path: str) -> str:
    """
    Converts an image file to a base64 data URI format that matches the extraction system.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Base64 data URI string (e.g., "data:image/png;base64,...")
    """
    import mimetypes
    
    # Get the MIME type based on file extension
    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type is None:
        # Default to PNG if we can't determine the type
        mime_type = "image/png"
    
    # Read and encode the image
    with open(image_path, "rb") as image_file:
        base64_data = base64.b64encode(image_file.read()).decode("utf-8")
    
    # Create proper data URI format
    return f"data:{mime_type};base64,{base64_data}"

def create_plot_extractor(cfg: DictConfig) -> ClaudeLinePlotDataExtractor:
    """Create plot extractor."""
    model_name = cfg.get("model_name", "claude-sonnet-4-20250514")
    # Create Claude extractor
    return ClaudeLinePlotDataExtractor(model_name=model_name)

def create_markdown_from_images(image_dir: str) -> str:
    """
    Create a markdown string with all images in a directory embedded as data URIs.
    
    Args:
        image_dir: Directory containing image files
        
    Returns:
        Markdown string with embedded images
    """
    markdown_lines = []
    
    # Get all image files
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    image_files = ["figure_0.png"]
    
    for image_filename in image_files:
        image_path = os.path.join(image_dir, image_filename)
        
        # Convert to data URI
        data_uri = image_to_base64_data_uri(image_path)
        
        # Create markdown image syntax
        alt_text = f"Figure {image_filename}"
        markdown_line = f"![{alt_text}]({data_uri})"
        
        # TODO: Add some context around the image
        markdown_lines.append(markdown_line)
    
    return "\n".join(markdown_lines), image_files

def load_figures_from_directory(image_dir: str, paper_text: str = "Synthetic data for evaluation.") -> tuple[list[FigureInfoWithPaper], list[str]]:
    """
    Load all figures from a directory using the existing markdown parsing infrastructure.
    
    Args:
        image_dir: Directory containing image files
        paper_text: Text to use as paper context
        
    Returns:
        List of FigureInfoWithPaper objects
    """
    # Create markdown with embedded images
    markdown_text, figure_names_list = create_markdown_from_images(image_dir)
    
    # Use existing function to extract figures
    figures = find_figures_in_markdown(markdown_text)
    
    # Convert to FigureInfoWithPaper
    figures_with_paper = []
    for figure in figures:
        figure_with_paper = FigureInfoWithPaper(
            base64_data=figure.base64_data,
            alt_text=figure.alt_text,
            position=figure.position,
            context_before=figure.context_before,
            context_after=figure.context_after,
            figure_reference=figure.figure_reference,
            figure_class="Scatter plot",
            quantitative=True,
            paper_text=paper_text,
            si_text="",
        )
        figures_with_paper.append(figure_with_paper)
    
    return figures_with_paper, figure_names_list

@hydra.main(config_path="../config", config_name="config.yaml", version_base=None)
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger(__name__)

    # Get labels and images, set in config.yaml
    image_dir = cfg.get("image_dir", os.path.join(base_dir, "examples/notebooks"))
    label_dir = cfg.get("label_dir", os.path.join(base_dir, "examples/notebooks"))

    # Instantiate components from Hydra config
    extractor: PlotDataExtractorInterface = create_plot_extractor(cfg)
    metric = instantiate(cfg.plot_extraction.metric)

    # Load all figures at once
    figures, figure_names_list = load_figures_from_directory(image_dir)
    
    results = []
    log.info(f"Found {len(figures)} plots to evaluate.")
    
    for i, (figure_info, image_filename) in enumerate(tqdm(zip(figures,figure_names_list), desc="Evaluating Plots")):
        # Extract filename from figure reference or alt text
        if "Figure " in image_filename:
            image_filename = image_filename.replace("Figure ", "")
        
        json_path = os.path.join(label_dir, image_filename.replace(".png", ".json"))
        
        if not os.path.exists(json_path):
            log.warning(f"Ground truth not found for {image_filename}, skipping.")
            continue
        
        try:
            # Run extraction and evaluation
            predicted_data = extractor.forward(figure_info)
            ground_truth_data = load_ground_truth(json_path)
            
            score = metric(predicted_data, ground_truth_data)
            
            results.append({"filename": image_filename, "score": score})
            log.info(f"Evaluated {image_filename}: Score = {score:.4f}")
            
        except Exception as e:
            log.error(f"Failed to process {image_filename}: {e}")
            results.append({"filename": image_filename, "score": 0.0})

    # Generate and save summary report
    if results:
        results_df = pd.DataFrame(results)
        output_path = "plot_evaluation_results.csv"
        results_df.to_csv(output_path, index=False)

        mean_score = results_df["score"].mean()
        std_dev = results_df["score"].std()

        log.info("\n--- EVALUATION SUMMARY ---")
        log.info(f"Total plots evaluated: {len(results_df)}")
        log.info(f"Mean Score: {mean_score:.4f}")
        log.info(f"Standard Deviation: {std_dev:.4f}")
        log.info(f"Results saved to: {os.path.abspath(output_path)}")


if __name__ == "__main__":
    main() 