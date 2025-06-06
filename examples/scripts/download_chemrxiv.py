import os

import chemrxiv
from datasets import load_dataset
from dotenv import load_dotenv
from tqdm import tqdm

from llm_synthesis.transformers.pdf_extraction import (
    DoclingPDFExtractor,
    MistralPDFExtractor,
)

load_dotenv()

# Configuration
DATA_DIR = "/Users/mlederbau/lematerial-llm-synthesis/data"
PDFS_DIR = os.path.join(DATA_DIR, "pdfs_chemrxiv")
HUGGINGFACE_DATASET = "magdaroni/chemrxiv-dev"
SPLIT = "train"
NUM_PAPERS = (
    5  # Change or remove this if you want to process more than the first two
)


def ensure_directory(path: str):
    """Create the directory if it doesn't already exist."""
    os.makedirs(path, exist_ok=True)


def download_pdf_by_doi(
    client: chemrxiv.Client, doi: str, out_dir: str, filename: str
) -> str:
    """
    Download a PDF from ChemRxiv given its DOI.
    Returns the full path to the downloaded file.
    """
    paper = client.item_by_doi(doi)
    paper.download_pdf(dirpath=out_dir, filename=filename)
    return os.path.join(out_dir, filename)


def download_si_by_doi(
    client: chemrxiv.Client, doi: str, out_dir: str, filename: str
) -> str:
    """
    Download a SI from ChemRxiv given its DOI.
    Returns the full path to the downloaded file.
    """
    paper = client.item_by_doi(doi)
    paper.download_si(dirpath=out_dir, filename=filename)
    return os.path.join(out_dir, filename)


def extract_text_from_pdf(
    extractor: DoclingPDFExtractor, pdf_path: str
) -> str:
    """
    Read raw PDF bytes from disk and run the Docling extractor.
    Returns the extracted markdown string.
    """
    with open(pdf_path, "rb") as f:
        raw_bytes = f.read()
    return extractor.extract(raw_bytes)


def main():
    # 1) Load the ChemRxiv dataset (train split)
    # and convert to a pandas DataFrame
    dataset = load_dataset(HUGGINGFACE_DATASET, split=SPLIT)
    df = dataset.to_pandas()

    # 2) Prepare ChemRxiv client and Docling extractor
    client = chemrxiv.Client()
    pdf_extractor = MistralPDFExtractor(structured=False)

    # 3) Ensure output directory exists
    ensure_directory(PDFS_DIR)

    # 4) Optionally take only the first N rows for faster testing
    df_to_process = df.head(NUM_PAPERS)

    # 5) Loop over each paper, download its PDF, and extract text
    for i, row in tqdm(df_to_process.iterrows(), total=len(df_to_process)):
        doi = row["doi"]
        paper_id = row["id"]
        filename = f"{paper_id}.pdf"
        filename_si = f"{paper_id}_si.pdf"

        # Download PDF
        pdf_path = download_pdf_by_doi(client, doi, PDFS_DIR, filename)

        try:
            si_path = download_si_by_doi(client, doi, PDFS_DIR, filename_si)
            markdown_text_si = extract_text_from_pdf(pdf_extractor, si_path)
        except Exception as e:
            print(f"Error downloading SI for paper ID {paper_id}: {e}")
            markdown_text_si = ""

        # Extract markdown text
        markdown_text = extract_text_from_pdf(pdf_extractor, pdf_path)

        df.loc[i, "text_paper"] = markdown_text
        df.loc[i, "text_si"] = markdown_text_si

    df.to_csv(f"{DATA_DIR}/chemrxiv_papers.csv", index=False)


if __name__ == "__main__":
    main()
