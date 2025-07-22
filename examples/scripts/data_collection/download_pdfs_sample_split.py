import asyncio
import os
import warnings
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pandas as pd
from datasets import DatasetDict, Value, load_dataset
from dotenv import load_dotenv
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

from llm_synthesis.transformers.pdf_extraction import (
    MistralPDFExtractor,
)

warnings.filterwarnings("ignore")
load_dotenv()

DATA_DIR = "/Users/mlederbau/lematerial-llm-synthesis/data/"
PDFS_DIR = os.path.join(
    DATA_DIR, "pdfs_for_sample_eval"
)  # Changed PDFS_DIR for this specific task
HUGGINGFACE_DATASET = "LeMaterial/LeMat-Synth"
SPLIT = "sample_for_evaluation"  # The split we want to update
BATCH_SIZE = 20


def ensure_directory(path: str):
    os.makedirs(path, exist_ok=True)


async def extract_text_from_pdf_async(
    extractor: MistralPDFExtractor, pdf_path: str
) -> str:
    with open(pdf_path, "rb") as f:
        return await extractor.aforward(f.read())


def download_file(
    url: str, dirpath: str = "./", filename: str = "file.pdf"
) -> str:
    """Private helper method to download a file from a URL."""
    out_path = os.path.join(dirpath, filename)

    # Create a Request with a browser-like User-Agent
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})

    # Open + read + write + close
    with urlopen(req) as response, open(out_path, "wb") as f:
        f.write(response.read())

    return out_path


async def process_paper_async(
    i: int,  # Original index in the filtered DataFrame/Dataset
    row: dict,  # Now accepting a dictionary (HF DS example format)
    pdf_extractor: MistralPDFExtractor,
    pdfs_dir: str,
) -> tuple[int, str, str]:  # Return original index, paper_text, si_text
    """
    Asynchronously downloads a PDF, extracts text, and returns results.
    Returns original index, paper_text, si_text (empty string if not available).
    """
    text_paper = None
    text_si = ""  # si_text remains empty string as per requirement

    try:
        pdf_url = row.get("pdf_url")
        paper_id = row.get("id")

        if not pdf_url:
            print(f"Skipping processing for ID {paper_id}: No 'pdf_url' found.")
            return i, None, None

        if not paper_id:
            print(f"Skipping processing for row {i}: No 'id' found.")
            return i, None, None

        pdf_path = await asyncio.to_thread(
            download_file, pdf_url, pdfs_dir, f"{paper_id}.pdf"
        )
    except HTTPError as e:
        print(f"Error downloading file for ID {paper_id}: {e}")
        return i, None, None
    except Exception as e:
        print(f"Unexpected error during download for ID {paper_id}: {e}")
        return i, None, None

    try:
        text_paper = await extract_text_from_pdf_async(pdf_extractor, pdf_path)
    except Exception as e:
        print(f"Error extracting text from PDF for ID {paper_id}: {e}")
        return i, None, None

    return i, text_paper, text_si


def update_and_push_split(
    processed_df: pd.DataFrame, original_dataset_dict: DatasetDict
):
    """
    Updates 'sample_for_evaluation' split with proc. data and pushes to Hub.
    This function is adapted for the specific update logic.
    """
    print("Updating 'sample_for_evaluation' split...")

    # Get the original 'sample_for_evaluation' split
    sample_eval_ds = original_dataset_dict[SPLIT]

    # Create lookup map for proc. data: id -> {"text_paper": .., "text_si": ..}
    # Ensure processed_df is clean of Nones before creating lookup
    processed_df_cleaned = processed_df.dropna(subset=["text_paper"]).set_index(
        "id"
    )
    processed_data_map = processed_df_cleaned[
        ["text_paper", "text_si"]
    ].to_dict(orient="index")

    # Define new features for text_paper and text_si
    updated_features = sample_eval_ds.features.copy()
    updated_features["text_paper"] = Value(dtype="string")
    updated_features["text_si"] = Value(dtype="string")

    def update_example(example):
        example_id = example.get("id")
        if (
            example_id in processed_data_map
            and example.get("source") == "omg24"
        ):
            processed_info = processed_data_map[example_id]
            example["text_paper"] = processed_info["text_paper"]
            example["text_si"] = processed_info["text_si"]  # This will be ""
        return example

    updated_sample_eval_ds = sample_eval_ds.map(
        update_example,
        features=updated_features,  # Apply the updated schema
        num_proc=os.cpu_count(),  # Use parallelism for speed
        load_from_cache_file=False,
        # Set to False to re-run map even if cached results exist
    )

    final_dataset_dict_to_push = DatasetDict()
    for split_name, ds in original_dataset_dict.items():
        if split_name == SPLIT:
            # If it's the target split, use the updated version
            final_dataset_dict_to_push[SPLIT] = updated_sample_eval_ds
        else:
            # Otherwise, copy the original split directly
            final_dataset_dict_to_push[split_name] = ds

    # Push to Hub
    print(f"Pushing updated '{SPLIT}' split to HF Hub: {HUGGINGFACE_DATASET}")
    # final_dataset_dict_to_push.push_to_hub(HUGGINGFACE_DATASET)
    print(
        f"→ Pushed {len(updated_sample_eval_ds)} records"
        f"to HF under split “{SPLIT}”"
    )


async def main_async():
    # 1) Load the entire DatasetDict from the Hub to preserve all splits
    original_dataset_dict = load_dataset(HUGGINGFACE_DATASET)
    # Get the specific split we want to work on
    sample_for_evaluation_ds = original_dataset_dict[SPLIT]

    print(
        f"Loaded '{SPLIT}' split with {len(sample_for_evaluation_ds)} entries."
    )

    # 2) Filter for 'omg24' source
    omg24_samples_to_process = sample_for_evaluation_ds.filter(
        lambda example: example.get("source") == "omg24",
        num_proc=os.cpu_count(),
        load_from_cache_file=False,
    )
    print(f"Found {len(omg24_samples_to_process)} 'omg24' samples to process.")

    if len(omg24_samples_to_process) == 0:
        print("No 'omg24' samples found in 'sample_for_evaluation'. Exiting.")
        return

    pdf_extractor = MistralPDFExtractor()
    ensure_directory(PDFS_DIR)

    processed_results = []
    tasks = []

    # 3) Schedule extraction tasks for 'omg24' samples
    df_to_process = omg24_samples_to_process.to_pandas()

    for i, row_dict in tqdm(
        df_to_process.iterrows(),
        total=len(df_to_process),
        desc="Scheduling PDF extraction tasks",
    ):
        tasks.append(process_paper_async(i, row_dict, pdf_extractor, PDFS_DIR))

        # 4) Process tasks in batches
        if len(tasks) >= BATCH_SIZE:
            results = await tqdm_asyncio.gather(
                *tasks, desc="Processing PDF Batch"
            )
            processed_results.extend(results)
            tasks = []

    # 5) Process any remaining tasks
    if tasks:
        results = await tqdm_asyncio.gather(
            *tasks, desc="Processing Last PDF Batch"
        )
        processed_results.extend(results)

    # 6) Convert results to a DataFrame for easier merging/lookup
    processed_df_for_update = pd.DataFrame(
        [
            {
                "id": df_to_process.loc[idx, "id"],
                "text_paper": tp,
                "text_si": tsi,
            }
            for idx, tp, tsi in processed_results
        ]
    )

    # 7) Update the 'sample_for_evaluation' split and push
    update_and_push_split(processed_df_for_update, original_dataset_dict)

    print("\nProcessing complete for 'sample_for_evaluation' (omg24 sources).")


if __name__ == "__main__":
    asyncio.run(main_async())
