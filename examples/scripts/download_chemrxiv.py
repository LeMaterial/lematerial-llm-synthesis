import os
import warnings

import chemrxiv
from datasets import Dataset, DatasetDict, load_dataset
from dotenv import load_dotenv
from tqdm import tqdm

from llm_synthesis.transformers.pdf_extraction import (
    DoclingPDFExtractor,
)

warnings.filterwarnings("ignore")
load_dotenv()

DATA_DIR = "/Users/mlederbau/lematerial-llm-synthesis/data"
PDFS_DIR = os.path.join(DATA_DIR, "pdfs_chemrxiv")
HUGGINGFACE_DATASET = "magdaroni/chemrxiv-dev"
SPLIT = "filtered_matsci"
BATCH_SIZE = 2


def ensure_directory(path: str):
    os.makedirs(path, exist_ok=True)


def download_pdf_by_doi(client, doi, out_dir, filename):
    paper = client.item_by_doi(doi)
    paper.download_pdf(dirpath=out_dir, filename=filename)
    return os.path.join(out_dir, filename)


def download_si_by_doi(client, doi, out_dir, filename):
    paper = client.item_by_doi(doi)
    paper.download_si(dirpath=out_dir, filename=filename)
    return os.path.join(out_dir, filename)


def extract_text_from_pdf(extractor, pdf_path):
    with open(pdf_path, "rb") as f:
        return extractor.extract(f.read())


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
    pdf_extractor = DoclingPDFExtractor(
        pipeline="standard",
        table_mode="accurate",
        add_page_images=False,
        use_gpu=True,
        scale=2.0,
        format="markdown",
    )
    ensure_directory(PDFS_DIR)

    processed = 0
    for i, row in tqdm(df.iterrows(), total=len(df)):
        # skip if already extracted
        if (
            df.loc[i, "text_paper"] is not None
            and df.loc[i, "text_si"] is not None
        ):
            continue

        doi, pid = row["doi"], row["id"]
        # download + extract paper
        pdf_path = download_pdf_by_doi(client, doi, PDFS_DIR, f"{pid}.pdf")
        text_paper = extract_text_from_pdf(pdf_extractor, pdf_path)

        # download + extract SI (if any)
        try:
            si_path = download_si_by_doi(
                client, doi, PDFS_DIR, f"{pid}_si.pdf"
            )
            text_si = extract_text_from_pdf(pdf_extractor, si_path)
        except Exception:
            text_si = ""

        df.at[i, "text_paper"] = text_paper
        df.at[i, "text_si"] = text_si

        processed += 1
        # every BATCH_SIZE papers, push
        if processed % BATCH_SIZE == 0:
            push_current_df(df)

    # final push for the tail
    if processed % BATCH_SIZE != 0:
        push_current_df(df)

    # also save locally
    df.to_csv(f"{DATA_DIR}/chemrxiv_papers.csv", index=False)


if __name__ == "__main__":
    main()
