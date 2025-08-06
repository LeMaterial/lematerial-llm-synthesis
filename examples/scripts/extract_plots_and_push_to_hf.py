import logging

import tqdm
from datasets import Dataset, load_dataset

from llm_synthesis.transformers.figure_extraction.hf_figure_extractor import (
    HFFigureExtractor,
)
from llm_synthesis.transformers.plot_extraction.claude_extraction.plot_data_extraction import (  # noqa: E501
    ClaudeLinePlotDataExtractor,
)

# only log info
logging.basicConfig(level=logging.INFO)


def main(batch_size: int = 10, config="default", split="sample_for_evaluation"):
    hf_figure_extractor = HFFigureExtractor()

    # Initialize the extractor
    extractor = ClaudeLinePlotDataExtractor(
        model_name="claude-sonnet-4-20250514"
    )

    dataset = load_dataset(
        "LeMaterial/LeMat-Synth", name=config, split=split
    )
    df = dataset.to_pandas()
    for idx, row in tqdm.tqdm(df.iterrows(), total=len(df)):
        logging.info(row.paper_doi)
        segmented_images = hf_figure_extractor.forward(row["images"])
        logging.info(f"Found {len(segmented_images)} figures in the paper.")
        quantative_images = [
            img for img in segmented_images if img.quantitative
        ]
        logging.info(
            f"Found {len(quantative_images)} quantitative figures in the paper."
        )
        line_charts = [
            img for img in quantative_images if img.figure_class == "Line plot"
        ]
        bar_charts = [
            img for img in quantative_images if img.figure_class == "Bar plot"
        ]
        scatter_plots = [
            img
            for img in quantative_images
            if img.figure_class == "Scatter plot"
        ]
        logging.info(f"Found {len(line_charts)} line charts in the paper.")
        logging.info(f"Found {len(bar_charts)} bar charts in the paper.")
        logging.info(f"Found {len(scatter_plots)} scatter plots in the paper.")

        plot_data = []

        # add to plot_data in row the bar_charts
        for line_chart in line_charts:
            extracted_data = extractor.forward(line_chart)
            extracted_data.figure_class = line_chart.figure_class
            logging.info(extracted_data)
            plot_data.append(extracted_data)
            logging.info("-" * 100)

        row["plot_data"] = plot_data
        if idx % batch_size == 0:
            logging.info(f"Pushing batch {idx // batch_size} to hub")
            ds = Dataset.from_pandas(df)
            ds.push_to_hub("LeMaterial/LeMat-Synth", config_name=config, split=split, create_pr=True)
            df = dataset.to_pandas()

    ds = Dataset.from_pandas(df)
    ds.push_to_hub("LeMaterial/LeMat-Synth", config_name=config, split=split, create_pr=True)


if __name__ == "__main__":
    main(batch_size=10)
