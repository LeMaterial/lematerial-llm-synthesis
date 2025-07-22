import json
import os

import requests
from datasets import load_dataset
from tqdm.auto import tqdm  # For a nice progress bar


def get_samples_to_annotate():
    """
    Loads samples from the "LeMaterial/LeMat-Synth" dataset, downloads
    associated PDFs, and saves structured synthesis entries and PDFs
    to the 'annotations/wip/' directory.
    """
    output_dir = "annotations/wip"
    os.makedirs(output_dir, exist_ok=True)

    print("Loading dataset 'LeMaterial/LeMat-Synth' split ")
    try:
        dataset = load_dataset(
            "LeMaterial/LeMat-Synth", split="sample_for_evaluation"
        )
        print(f"Dataset loaded successfully with {len(dataset)} samples.")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Please ensure internet connectivity and the dataset exists.")
        return

    print("Processing samples and downloading PDFs...")
    for sample in tqdm(dataset, desc="Processing samples"):
        sample_id = sample.get("id")
        pdf_url = sample.get("pdf_url")
        structured_synthesis = sample.get("structured_synthesis")

        if not sample_id:
            print("Warning: Sample found without 'id'. Skipping.")
            continue
        if not pdf_url:
            print(f"ID {sample_id} has no 'pdf_url'. Skipping PDF.")
        if structured_synthesis is None:
            print(
                f"ID {sample_id} has no 'structured_synthesis'. Skipping JSON."
            )

        if structured_synthesis is not None:
            json_filename = os.path.join(output_dir, f"{sample_id}.json")
            try:
                with open(json_filename, "w", encoding="utf-8") as f:
                    json.dump(
                        structured_synthesis, f, ensure_ascii=False, indent=4
                    )
            except Exception as e:
                print(f"Error saving JSON for ID {sample_id}: {e}")

        if pdf_url:
            pdf_filename = os.path.join(output_dir, f"{sample_id}.pdf")
            try:
                response = requests.get(pdf_url, stream=True)
                response.raise_for_status()
                with open(pdf_filename, "wb") as pdf_file:
                    for chunk in response.iter_content(chunk_size=8192):
                        pdf_file.write(chunk)
            except requests.exceptions.RequestException as e:
                print(
                    f"Error downloading PDF ID {sample_id} from {pdf_url}: {e}"
                )
            except Exception as e:
                print(f"Error occurred downloading PDF for ID {sample_id}: {e}")

    print(f"\nFinished processing samples. Files saved to '{output_dir}'.")


if __name__ == "__main__":
    get_samples_to_annotate()
