import asyncio
import logging
import os
import warnings

import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset
from dotenv import load_dotenv
from playwright.async_api import async_playwright
from tqdm.asyncio import tqdm_asyncio

# Assuming you have this extractor from your previous setup
from llm_synthesis.transformers.pdf_extraction import MistralPDFExtractor

warnings.filterwarnings("ignore")
load_dotenv()
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
SOURCE_REPO = "LeMaterial/LeMat-Synth-Papers"
SUBSET = "full"
SPLIT = "omg24"

# Local directories
USER = os.environ.get("USER", "your-local-user")
DATA_DIR = "/Users/magdalenalederbauer/Code/lematerial-llm-synthesis/data/"
PDFS_DIR = os.path.join(DATA_DIR, "pdfs_omg24_fix")

# Processing
BATCH_SIZE = 11000  # Process and push in batches of 10


def ensure_directory(path: str):
    """Creates a directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


async def download_file_with_playwright(
    browser, url: str, out_path: str
) -> str:
    """
    Downloads a file using Playwright, correctly handling direct download
    links by saving the file even when a navigation "error" occurs.
    """
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return out_path

    context = None
    page = None
    try:
        context = await browser.new_context(accept_downloads=True)
        page = await context.new_page()

        # The `expect_download` context manager will wait for a download.
        async with page.expect_download() as download_info:
            # We now put the navigation attempt inside its own try block.
            try:
                # Attempt to navigate. This will likely trigger the "error".
                await page.goto(url)
            except Exception as e:
                # We check if the error is the one we expect.
                if "Download is starting" in str(e):
                    # If it is, we log it and continue. The download is
                    # being handled by the `download_info` context.
                    logging.info(
                        f"Download triggered for {os.path.basename(out_path)}"
                    )
                else:
                    # If it's a different error, we raise it.
                    raise e

        # This code now runs AFTER the navigation attempt.
        # The download_info object now contains the captured download.
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
    """Asynchronously extracts text from a PDF file."""
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
    """Processes a single row: downloads PDF and extracts text."""
    pdf_path = os.path.join(PDFS_DIR, f"{row['id']}.pdf")

    # 1. Download the PDF
    downloaded_path = await download_file_with_playwright(
        browser, row["pdf_url"], pdf_path
    )

    # 2. Extract text from the downloaded PDF
    extracted_text = await extract_text_from_pdf_async(
        pdf_extractor, downloaded_path
    )

    return row.name, extracted_text  # Return DataFrame index and the text


def push_updates_to_hub(df: pd.DataFrame, features):
    """Pushes the updated DataFrame to the Hugging Face Hub."""
    logging.info(f"\nPreparing to push {len(df)} records to the Hub...")

    # Convert the updated pandas DataFrame back to a Dataset
    updated_dataset = Dataset.from_pandas(df, features=features)

    # Create a DatasetDict with the updated split
    # This will overwrite the old split with your new one when pushed
    final_ds_dict = DatasetDict({SPLIT: updated_dataset})

    # Push to hub, creating a pull request
    try:
        final_ds_dict.push_to_hub(
            SOURCE_REPO,
            commit_message=f"Update {SPLIT} with extracted text",
            create_pr=True,
        )
        logging.info(f"Successfully pushed updates to {SOURCE_REPO}.")
    except Exception as e:
        logging.info(f"Failed to push to Hub: {e}")


async def main():
    """Main function to run the data fixing pipeline."""
    ensure_directory(PDFS_DIR)

    # 1. Load the specific dataset split
    logging.info(
        f"Loading dataset '{SOURCE_REPO}', subset '{SUBSET}', split '{SPLIT}'"
    )
    ds = load_dataset(SOURCE_REPO, name=SUBSET, split=SPLIT)
    df = ds.to_pandas()

    original_features = ds.features

    # 2. Filter for rows that need processing
    df_to_process = df[
        pd.isna(df["text_paper"]) | (df["text_paper"] == "")
    ].copy()

    logging.info(
        f"Found {len(df_to_process)} records with missing",
        "'text_paper' ({len(df_to_process) / len(df) * 100:.2f}%).",
    )

    # 3. Initialize tools
    pdf_extractor = MistralPDFExtractor()

    async with async_playwright() as p:
        browser = await p.webkit.launch(headless=True)
        tasks = []

        # 4. Create and process tasks in batches
        for _, row in df_to_process.iterrows():
            tasks.append(process_row_async(row, browser, pdf_extractor))

            if len(tasks) >= BATCH_SIZE:
                results = await tqdm_asyncio.gather(
                    *tasks, desc=f"Processing Batch (size {BATCH_SIZE})"
                )
                for index, text in results:
                    if text:
                        df.loc[index, "text_paper"] = text
                # Push the updated full dataframe after each batch
                push_updates_to_hub(df, original_features)
                tasks = []

        # 5. Process any remaining tasks in the last batch
        if tasks:
            results = await tqdm_asyncio.gather(
                *tasks, desc="Processing Final Batch"
            )
            for index, text in results:
                if text:
                    df.loc[index, "text_paper"] = text
            # Final push for the remainder
            push_updates_to_hub(df, original_features)

    logging.info("Success.")


if __name__ == "__main__":
    # It's recommended to log in to Hugging Face CLI first
    # huggingface-cli login
    asyncio.run(main())
