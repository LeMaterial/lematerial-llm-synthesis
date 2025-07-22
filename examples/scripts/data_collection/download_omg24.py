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
HUGGINGFACE_DATASET = "magdaroni/chemrxiv-dev"
SPLIT = "filtered_omg24"
BATCH_SIZE = 20


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
        print(f"Error downloading file: {e}")
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


def push_current_df(df_clean, orig, matsci_feats):
    # drop rows that failed
    df_clean = df_clean.dropna(subset=["text_paper"]).reset_index(drop=True)

    # convert with the same schema as filtered_matsci
    ds_new = Dataset.from_pandas(df_clean, features=matsci_feats)

    merged = DatasetDict(
        {
            **orig,  # keeps filtered_matsci + old split
            SPLIT: ds_new,  # overrides filtered_omg24 with your new one
        }
    )
    merged.push_to_hub(HUGGINGFACE_DATASET)
    print(f"→ Pushed {len(df_clean)} records to HF under split “{SPLIT}”")


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


def main():
    # Login using e.g. `huggingface-cli login` to access this dataset
    df = load_dataset("iknow-lab/open-materials-guide-2024")[
        "train"
    ].to_pandas()
    orig = load_dataset(HUGGINGFACE_DATASET)
    matsci_feats = orig["filtered_matsci"].features

    df_new = pd.DataFrame(
        columns=[
            "id",
            "title",
            "authors",
            "abstract",
            "doi",
            "published_date",
            "updated_date",
            "categories",
            "license",
            "pdf_url",
            "views_count",
            "read_count",
            "citation_count",
            "keywords",
            "text_paper",
            "text_si",
        ]
    )
    df_new["id"] = df["id"]
    df_new["title"] = df["title"]
    df_new["authors"] = df["authors"].apply(lambda x: str(x))
    df_new["abstract"] = df["abstract"]
    df_new["doi"] = None
    df_new["published_date"] = None
    df_new["updated_date"] = None
    df_new["categories"] = None
    df_new["license"] = None
    df_new["pdf_url"] = df["pdf_url"]
    df_new["views_count"] = None
    df_new["read_count"] = None
    df_new["citation_count"] = None
    df_new["keywords"] = None
    df_new["text_paper"] = None
    df_new["text_si"] = None

    pdf_extractor = MistralPDFExtractor()
    ensure_directory(PDFS_DIR)

    processed = 0

    for i, row in tqdm(df_new.iterrows(), total=len(df_new)):
        # skip if already extracted
        if row["text_paper"] is not None:
            continue

        text_paper, text_si = process_paper(row, pdf_extractor, PDFS_DIR)

        df_new.at[i, "text_paper"] = text_paper
        df_new.at[i, "text_si"] = text_si

        processed += 1
        # every BATCH_SIZE papers, push
        if processed % BATCH_SIZE == 0:
            push_current_df(df_new, orig, matsci_feats)

    # final push for the tail
    if processed % BATCH_SIZE != 0:
        push_current_df(df_new, orig)

    df_new.to_csv(f"{DATA_DIR}/omg24_papers.csv", index=False)


async def main_async():
    # 1) load source data & existing hub splits
    df = load_dataset("iknow-lab/open-materials-guide-2024")[
        "train"
    ].to_pandas()
    orig = load_dataset(HUGGINGFACE_DATASET)
    matsci_feats = orig["filtered_matsci"].features

    # 2) initialize df_new with identical columns & defaults
    df_new = pd.DataFrame(
        columns=[
            "id",
            "title",
            "authors",
            "abstract",
            "doi",
            "published_date",
            "updated_date",
            "categories",
            "license",
            "pdf_url",
            "views_count",
            "read_count",
            "citation_count",
            "keywords",
            "text_paper",
            "text_si",
        ]
    )
    df_new["id"] = df["id"]
    df_new["title"] = df["title"]
    df_new["authors"] = df["authors"].apply(str)
    df_new["abstract"] = df["abstract"]
    df_new[
        [
            "doi",
            "published_date",
            "updated_date",
            "categories",
            "license",
            "views_count",
            "read_count",
            "citation_count",
            "keywords",
        ]
    ] = None
    df_new["pdf_url"] = df["pdf_url"]
    df_new["text_paper"] = None
    df_new["text_si"] = None

    pdf_extractor = MistralPDFExtractor()
    ensure_directory(PDFS_DIR)

    processed = 0
    tasks = []

    # 3) schedule all extractions as tasks
    for i, row in tqdm(df_new.iterrows(), total=len(df_new)):
        if row["text_paper"] is not None:
            continue
        tasks.append(process_paper_async(i, row, pdf_extractor, PDFS_DIR))

        # 4) once we hit a batch, await and push
        if len(tasks) >= BATCH_SIZE:
            results = await tqdm_asyncio.gather(*tasks, desc="Processing Batch")
            for j, text_paper, text_si in results:
                df_new.at[j, "text_paper"] = text_paper
                df_new.at[j, "text_si"] = text_si
            push_current_df(df_new, orig, matsci_feats)
            processed += len(tasks)
            tasks = []

    # 5) remaining tasks
    if tasks:
        results = await tqdm_asyncio.gather(
            *tasks, desc="Processing Last Batch"
        )
        for j, text_paper, text_si in results:
            df_new.at[j, "text_paper"] = text_paper
            df_new.at[j, "text_si"] = text_si
        push_current_df(df_new, orig, matsci_feats)
        processed += len(tasks)

    # 6) write out the full CSV locally
    df_new.to_csv(f"{DATA_DIR}/omg24_papers.csv", index=False)


if __name__ == "__main__":
    asyncio.run(main_async())
