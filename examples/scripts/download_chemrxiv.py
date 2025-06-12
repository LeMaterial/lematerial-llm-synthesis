import asyncio
import os
import warnings

import chemrxiv
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

DATA_DIR = "/home/gregoire/entalpic/lematerial-llm-synthesis/data"
PDFS_DIR = os.path.join(DATA_DIR, "pdfs_chemrxiv")
HUGGINGFACE_DATASET = "magdaroni/chemrxiv-dev"
SPLIT = "filtered_matsci"
BATCH_SIZE = 100


def ensure_directory(path: str):
    os.makedirs(path, exist_ok=True)


def download_pdf_by_doi(
    client: chemrxiv.Client, doi: str, out_dir: str, filename: str
) -> str:
    paper = client.item_by_doi(doi)
    paper.download_pdf(dirpath=out_dir, filename=filename)
    return os.path.join(out_dir, filename)


def download_si_by_doi(client, doi, out_dir, filename):
    paper = client.item_by_doi(doi)
    paper.download_si(dirpath=out_dir, filename=filename)
    return os.path.join(out_dir, filename)


async def extract_text_from_pdf_async(
    extractor: MistralPDFExtractor, pdf_path: str
) -> str:
    with open(pdf_path, "rb") as f:
        return await extractor.aforward(f.read())


def extract_text_from_pdf(
    extractor: MistralPDFExtractor, pdf_path: str
) -> str:
    with open(pdf_path, "rb") as f:
        return extractor.forward(f.read())


def process_paper(
    row: pd.Series,
    client: chemrxiv.Client,
    pdf_extractor: MistralPDFExtractor,
    pdfs_dir: str,
) -> tuple[str, str]:
    doi, pid = row["doi"], row["id"]
    pdf_path = download_pdf_by_doi(client, doi, pdfs_dir, f"{pid}.pdf")
    text_paper = extract_text_from_pdf(pdf_extractor, pdf_path)

    # Download SI in a thread (if not async)
    try:
        si_path = download_si_by_doi(client, doi, pdfs_dir, f"{pid}_si.pdf")
        text_si = extract_text_from_pdf(pdf_extractor, si_path)
    except Exception:
        text_si = ""

    return text_paper, text_si


async def process_paper_async(
    i: int,
    row: pd.Series,
    client: chemrxiv.Client,
    pdf_extractor: MistralPDFExtractor,
    pdfs_dir: str,
    semaphore: asyncio.Semaphore,
) -> tuple[str, str]:
    doi, pid = row["doi"], row["id"]
    # Download PDF in a thread (if not async)
    async with semaphore:
        pdf_path = await asyncio.to_thread(
            download_pdf_by_doi, client, doi, pdfs_dir, f"{pid}.pdf"
        )
        text_paper = await extract_text_from_pdf_async(pdf_extractor, pdf_path)

        # Download SI in a thread (if not async)
        try:
            si_path = await asyncio.to_thread(
                download_si_by_doi, client, doi, pdfs_dir, f"{pid}_si.pdf"
            )
            text_si = await extract_text_from_pdf_async(pdf_extractor, si_path)
        except Exception:
            text_si = ""

    return i, text_paper, text_si


def push_current_df(df):
    # convert to Dataset and push
    ds = Dataset.from_pandas(df.reset_index(drop=True))
    DatasetDict({SPLIT: ds}).push_to_hub(HUGGINGFACE_DATASET)
    print(f"→ Pushed {len(df)} records to HuggingFace under split “{SPLIT}”")


def main():
    orig = load_dataset(HUGGINGFACE_DATASET, split=SPLIT)
    df = orig.to_pandas()

    # filter by categories (as before)…

    categories = [
        "Solid State Chemistry",
        "Solution Chemistry",
        "Solvates",
        "Spectroscopy (Inorg.)",
        "Structure",
        "Supramolecular Chemistry (Inorg.)",
        "Supramolecular Chemistry (Org.)",
        "Surface",
        "Surfactants",
        "Thermal Conductors and Insulators",
        "Thin Films",
        "Wastes",
        "Water Purification",
    ]
    df = df[
        df["categories"].apply(lambda x: any(cat in x for cat in categories))
    ]

    client = chemrxiv.Client()
    pdf_extractor = MistralPDFExtractor()
    ensure_directory(PDFS_DIR)

    processed = 0

    for i, row in tqdm(df.iterrows(), total=len(df)):
        # skip if already extracted
        if row["text_paper"] is not None:
            continue

        text_paper, text_si = process_paper(
            row, client, pdf_extractor, PDFS_DIR
        )
        df.at[i, "text_paper"] = text_paper
        df.at[i, "text_si"] = text_si

        processed += 1
        # every BATCH_SIZE papers, push
        if processed % BATCH_SIZE == 0:
            push_current_df(df)

    # final push for the tail
    if processed % BATCH_SIZE != 0:
        push_current_df(df)

    df.to_csv(f"{DATA_DIR}/chemrxiv_papers.csv", index=False)


async def main_async():
    semaphore = asyncio.Semaphore(BATCH_SIZE)

    orig = load_dataset(HUGGINGFACE_DATASET, split=SPLIT)
    df = orig.to_pandas()

    # filter by categories (as before)…
    categories = [
        "Solid State Chemistry",
        "Solution Chemistry",
        "Solvates",
        "Spectroscopy (Inorg.)",
        "Structure",
        "Supramolecular Chemistry (Inorg.)",
        "Supramolecular Chemistry (Org.)",
        "Surface",
        "Surfactants",
        "Thermal Conductors and Insulators",
        "Thin Films",
        "Wastes",
        "Water Purification",
    ]
    df = df[
        df["categories"].apply(lambda x: any(cat in x for cat in categories))
    ]

    client = chemrxiv.Client()
    pdf_extractor = MistralPDFExtractor()
    ensure_directory(PDFS_DIR)

    tasks = []
    for i, row in tqdm(df.iterrows(), total=len(df)):
        # skip if already extracted
        if row["text_paper"] is not None:
            continue

        tasks.append(
            process_paper_async(
                i, row, client, pdf_extractor, PDFS_DIR, semaphore
            )
        )

    if len(tasks) > 0:
        results = await tqdm_asyncio.gather(
            *tasks, desc="Processing Last Batch"
        )
        for j, text_paper, text_si in results:
            df.at[j, "text_paper"] = text_paper
            df.at[j, "text_si"] = text_si
        push_current_df(df)
    # also save locally
    df.to_csv(f"{DATA_DIR}/chemrxiv_papers.csv", index=False)


if __name__ == "__main__":
    asyncio.run(main_async())
