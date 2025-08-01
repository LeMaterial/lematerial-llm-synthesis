import asyncio
import logging
import os
import warnings

import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset
from dotenv import load_dotenv
from playwright.async_api import async_playwright
from tqdm import tqdm

from llm_synthesis.transformers.pdf_extraction import MistralPDFExtractor

warnings.filterwarnings("ignore")
load_dotenv()
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
SOURCE_REPO = "LeMaterial/LeMat-Synth-Papers"
# SUBSET = "full"
# SPLIT = "omg24"
SUBSET = "default"
SPLIT = "sample_for_evaluation"


DATA_DIR = "/Users/magdalenalederbauer/Code/lematerial-llm-synthesis/data/"
PDFS_DIR = os.path.join(DATA_DIR, "pdfs_omg24_fix")

# Processing
BATCH_SIZE = 10  # A reasonable batch size for parallel processing


def ensure_directory(path: str):
    os.makedirs(path, exist_ok=True)


async def download_file_with_playwright(
    browser, url: str, out_path: str
) -> str:
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return out_path
    context = None
    page = None
    try:
        context = await browser.new_context(accept_downloads=True)
        page = await context.new_page()
        async with page.expect_download() as download_info:
            try:
                await page.goto(url)
            except Exception as e:
                if "Download is starting" in str(e):
                    logging.info(
                        f"Download triggered for {os.path.basename(out_path)}"
                    )
                else:
                    raise e
        download = await download_info.value
        await download.save_as(out_path)
        logging.info(f"Successfully SAVED: {os.path.basename(out_path)}")
        return out_path
    except Exception as e:
        logging.error(f"Download process FAILED for {url}: {e}")
        return None
    finally:
        if page and not page.is_closed():
            await page.close()
        if context:
            await context.close()


async def extract_text_from_pdf_async(
    extractor: MistralPDFExtractor, pdf_path: str
) -> str:
    if not pdf_path or not os.path.exists(pdf_path):
        return ""
    try:
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()
        return await extractor.aforward(pdf_bytes)
    except Exception as e:
        logging.info(f"Error extracting text from {pdf_path}: {e}")
        return ""


async def process_row_async(
    row: pd.Series, browser, pdf_extractor: MistralPDFExtractor
) -> tuple[int, str]:
    pdf_path = os.path.join(PDFS_DIR, f"{row['id']}.pdf")
    downloaded_path = await download_file_with_playwright(
        browser, row["pdf_url"], pdf_path
    )
    extracted_text = await extract_text_from_pdf_async(
        pdf_extractor, downloaded_path
    )
    return row.name, extracted_text


def push_updates_to_hub(df: pd.DataFrame, ds_dict: DatasetDict, features):
    """
    Safely updates a specific split within the full DatasetDict and pushes
    the entire configuration to the Hub, preserving all other splits.
    """
    logging.info(f"\nPreparing to push updates for split '{SPLIT}'...")

    # Convert the updated pandas DataFrame back to a Dataset
    updated_dataset = Dataset.from_pandas(df, features=features)

    # Replace the old split in the full dict, don't create a new one
    ds_dict[SPLIT] = updated_dataset

    # Push the complete, updated DatasetDict to the hub
    try:
        ds_dict.push_to_hub(
            SOURCE_REPO,
            commit_message=f"Update split '{SPLIT}' with extracted text",
            create_pr=True,
            config_name=SUBSET,
        )
        logging.info(f"Successfully pushed updates to {SOURCE_REPO}.")
    except Exception as e:
        logging.error(f"Failed to push to Hub: {e}")


async def main():
    """Main function to run the data fixing pipeline."""
    ensure_directory(PDFS_DIR)

    logging.info(
        f"Loading dataset configuration '{SOURCE_REPO}', subset '{SUBSET}'..."
    )
    ds_dict = load_dataset(SOURCE_REPO, name=SUBSET)

    df = ds_dict[SPLIT].to_pandas()
    original_features = ds_dict[SPLIT].features

    df_to_process = df[
        pd.isna(df["text_paper"]) | (df["text_paper"] == "")
    ].copy()

    logging.info(f"Found {len(df_to_process)} records to process")
    logging.info(f"({len(df_to_process) / len(df) * 100:.2f}%).")

    pdf_extractor = MistralPDFExtractor()

    async with async_playwright() as p:
        browser = await p.webkit.launch(headless=True)
        tasks = []

        for _, row in tqdm(
            df_to_process.iterrows(),
            total=len(df_to_process),
            desc="Processing files",
        ):
            tasks.append(process_row_async(row, browser, pdf_extractor))

            if len(tasks) >= BATCH_SIZE:
                results = await asyncio.gather(*tasks)
                for index, text in results:
                    if text:
                        df.loc[index, "text_paper"] = text
                tasks = []

        if tasks:
            results = await asyncio.gather(*tasks)
            for index, text in results:
                if text:
                    df.loc[index, "text_paper"] = text

    logging.info("All local processing finished. Starting final push to Hub...")
    push_updates_to_hub(df, ds_dict, original_features)

    logging.info("Success.")


if __name__ == "__main__":
    asyncio.run(main())
