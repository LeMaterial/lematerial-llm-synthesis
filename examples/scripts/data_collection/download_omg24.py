import asyncio
import os
import warnings
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset
from dotenv import load_dotenv
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

from llm_synthesis.transformers.pdf_extraction import (
    MistralPDFExtractor,
)

warnings.filterwarnings("ignore")
load_dotenv()

DATA_DIR = "/Users/mlederbau/lematerial-llm-synthesis/data/"
PDFS_DIR = os.path.join(DATA_DIR, "pdfs_omg24")
HUGGINGFACE_DATASET = "LeMaterial/LeMat-Synth"
SPLIT = "omg24"
BATCH_SIZE = 50


def ensure_directory(path: str):
    os.makedirs(path, exist_ok=True)


async def extract_text_from_pdf_async(
    extractor: MistralPDFExtractor, pdf_path: str
) -> str:
    with open(pdf_path, "rb") as f:
        return await extractor.aforward(f.read())


def extract_text_from_pdf(extractor: MistralPDFExtractor, pdf_path: str) -> str:
    with open(pdf_path, "rb") as f:
        return extractor.forward(f.read())


def process_paper(
    row: pd.Series,
    pdf_extractor: MistralPDFExtractor,
    pdfs_dir: str,
) -> tuple[str, str]:
    try:
        pdf_path = download_file(row["pdf_url"], pdfs_dir, f"{row['id']}.pdf")
    except HTTPError as e:
        print(f"Error downloading file: {e}, {row['pdf_url']}")
        return None, None

    try:
        text_paper = extract_text_from_pdf(pdf_extractor, pdf_path)
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return None, None

    text_si = ""

    return text_paper, text_si


async def process_paper_async(
    i: int,
    row: pd.Series,
    pdf_extractor: MistralPDFExtractor,
    pdfs_dir: str,
) -> tuple[str, str]:
    try:
        pdf_path = await asyncio.to_thread(
            download_file, row["pdf_url"], pdfs_dir, f"{row['id']}.pdf"
        )
    except HTTPError as e:
        print(f"Error downloading file: {e}")
        return i, None, None

    try:
        text_paper = await extract_text_from_pdf_async(pdf_extractor, pdf_path)
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return i, None, None

    text_si = ""

    return i, text_paper, text_si


def push_current_df(df_clean, orig, target_split_features):
    # drop rows that failed (where text_paper is None)
    df_to_push = df_clean.dropna(subset=["text_paper"]).reset_index(drop=True)

    # Convert to Dataset using the features of the target split
    # This is crucial for matching the schema exactly
    ds_new = Dataset.from_pandas(df_to_push, features=target_split_features)

    merged = DatasetDict(
        {
            **orig,  # keeps all other existing splits
            SPLIT: ds_new,  # overrides 'omg24' with your updated one
        }
    )
    merged.push_to_hub(HUGGINGFACE_DATASET)
    print(f"→ Pushed {len(df_to_push)} records to HF under split “{SPLIT}”")


def download_file(
    url: str, dirpath: str = "./", filename: str = "file.pdf"
) -> str:
    """Private helper method to download a file from a URL."""
    out_path = os.path.join(dirpath, filename)

    # Create a Request with a browser-like User-Agent
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})

    # Open + read + write to disk
    with urlopen(req) as response, open(out_path, "wb") as f:
        f.write(response.read())

    return out_path


async def main_async():
    # 1) Load the target dataset and specifically the 'omg24' split
    orig = load_dataset(HUGGINGFACE_DATASET)

    if SPLIT not in orig:
        raise ValueError(
            f"The split '{SPLIT}' does not exist in {HUGGINGFACE_DATASET}. "
            "Please ensure the 'omg24' split is created with the desired "
            "schema before running this script to populate text fields."
        )

    # Load the DataFrame directly from the 'omg24' split
    df_target_split = orig[SPLIT].to_pandas()
    print(
        f"Loaded {len(df_target_split)} records"
        f"from {HUGGINGFACE_DATASET}/{SPLIT}."
    )

    # Get the features of the 'omg24' split to ensure schema consistency on push
    omg24_features = orig[SPLIT].features

    df_new = df_target_split.copy()

    if "text_paper" not in df_new.columns:
        df_new["text_paper"] = None
    if "text_si" not in df_new.columns:
        df_new["text_si"] = None

    pdf_extractor = MistralPDFExtractor()
    ensure_directory(PDFS_DIR)

    processed = 0
    tasks = []

    # 3) Schedule all extractions as tasks
    for i, row in tqdm(df_new.iterrows(), total=len(df_new)):
        # Skip if 'text_paper' is already populated and not empty
        if pd.notna(row["text_paper"]) and str(row["text_paper"]).strip() != "":
            continue

        # Ensure 'pdf_url' exists for processing
        if "pdf_url" not in row or pd.isna(row["pdf_url"]):
            print(f"Skipping row {row['id']} due to missing pdf_url.")
            continue

        tasks.append(process_paper_async(i, row, pdf_extractor, PDFS_DIR))

        # 4) Once we hit a batch, await and push
        if len(tasks) >= BATCH_SIZE:
            results = await tqdm_asyncio.gather(*tasks, desc="Processing Batch")
            for j, text_paper, text_si in results:
                if text_paper is not None:
                    df_new.at[j, "text_paper"] = text_paper
                    df_new.at[j, "text_si"] = text_si

            # Pass the features of the target split to push_current_df
            push_current_df(df_new, orig, omg24_features)
            processed += len(tasks)
            tasks = []

    # 5) Remaining tasks
    if tasks:
        results = await tqdm_asyncio.gather(
            *tasks, desc="Processing Last Batch"
        )
        for j, text_paper, text_si in results:
            if text_paper is not None:
                df_new.at[j, "text_paper"] = text_paper
                df_new.at[j, "text_si"] = text_si

        push_current_df(df_new, orig, omg24_features)
        processed += len(tasks)

    # 6) Write out the full CSV locally
    df_new.to_csv(f"{DATA_DIR}/omg24_papers.csv", index=False)


if __name__ == "__main__":
    asyncio.run(main_async())
