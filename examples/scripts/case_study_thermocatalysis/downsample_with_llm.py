"""
Script to apply LLM filtering to papers identified by keyword search.
Loads paper IDs from pkl file and applies LLM inference to check for
performance vs temperature plots.
"""

import argparse
import os
import pickle

import datasets
import openai
import requests
from datasets import Features, Value, concatenate_datasets
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm
from transformers import AutoTokenizer

# --- Constants ---
MODEL_NAME = "mistralai/Ministral-3-14B-Instruct-2512"
MAX_MODEL_LEN = 50000
VLLM_ENDPOINT = "http://localhost:8000/v1"
OUTPUT_REPO = "amayuelas/LeMat-Synth-Papers-Catalysis-Filtered-v3"
DOWNLOAD_FOLDER = "../data/catalysis_pdfs-filtered-v3"
PKL_FILE = "results/db_thermocatalysis.pkl"

# --- LLM Prompt ---
PROMPT = """You are provided with a scientific materials paper.
We want to know if the paper contains a plot of material performance as a
function of temperature.
Read the paper carefully and determine if it contains a plot of material
performance vs temperature.

Answer with only yes or no.
Do not include any other text in your answer.
If you are not sure, answer with no.

Start Example:
Paper: [paper_text]
Question: Does this paper contain a plot of material performance as a function
of temperature?
Answer: [yes/no]
End Example.

Paper: {paper_text}
Question: Does this paper contain a plot of material performance as a function
of temperature?
Answer:
"""

# --- LLM Prompt ---
PROMPT_LONG = """You are a scientific paper analyzer specializing in
heterogeneous catalysis research.

Your task: Determine if this paper contains QUANTITATIVE line/curve plots
showing catalytic performance as a function of temperature.

REQUIRED CRITERIA (all must be met):
1. The plot must show TEMPERATURE on one axis (typically x-axis, in °C or K)
2. The plot must show at least ONE catalytic performance metric
on the other axis:
   - Conversion (%) or conversion rate
   - Selectivity (%)
   - Yield (%)
   - Turnover frequency (TOF) or turnover number (TON)
   - Reaction rate or activity
   - Product formation rate
3. The plot must be a LINE CHART or CURVE
(showing trends across multiple temperatures)
4. The plot must show EXPERIMENTAL catalytic data
(not just computational predictions)

EXCLUDE papers that only have:
- Single-point temperature measurements (no trend)
- TGA (thermogravimetric analysis) curves
- DSC (differential scanning calorimetry) curves
- Temperature-programmed desorption/reduction (TPD/TPR) profiles
- Stability tests at fixed temperature
- Arrhenius plots (these are ln(rate) vs 1/T, not performance vs T)

Examples of VALID plots:
- "Conversion vs temperature" showing how CO2 conversion changes from 200-500°C
- "Selectivity to methanol vs reaction temperature" 
- "Catalytic activity vs temperature" with multiple temperature points

Answer ONLY with "yes" or "no". No other text.
If uncertain, answer "no".

Paper: {paper_text}

Does this paper contain a quantitative line/curve plot of catalytic
performance vs temperature?
Answer:"""


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
)
def ask_llm_is_plot_of_material_performance_vs_temperature(
    text, client, model, selected_prompt
):
    message = selected_prompt.format(paper_text=text)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": message}],
            temperature=0,
            max_tokens=100,
        )
        answer = response.choices[0].message.content.strip().lower()
        return answer in ["yes", "yes."] or ("yes" in answer)
    except Exception as e:
        print(f"LLM call failed: {e}")
        return False


def process_example(example, client, model, tokenizer, selected_prompt):
    text = example["text_paper"]
    tokens = tokenizer.encode(text)
    if len(tokens) > MAX_MODEL_LEN:
        text = tokenizer.decode(tokens[: MAX_MODEL_LEN - 150])
    return ask_llm_is_plot_of_material_performance_vs_temperature(
        text, client, model, selected_prompt
    )


# --- Main Workflow ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt", choices=["default", "long"], default="default"
    )
    args = parser.parse_args()

    selected_prompt = PROMPT if args.prompt == "default" else PROMPT_LONG
    print(f"Using {args.prompt} prompt\n")

    # Load keyword search results from pkl
    print(f"Loading keyword search results from {PKL_FILE}...")
    with open(PKL_FILE, "rb") as f:
        keyword_db = pickle.load(f)

    # Flatten all IDs from keyword search across all splits
    keyword_ids = set()
    for split_name, ids in keyword_db.items():
        keyword_ids.update(ids)
        print(f"  {split_name}: {len(ids)} papers")
    print(f"Total unique papers from keyword search: {len(keyword_ids)}")

    # Load full dataset
    print("\nLoading LeMat-Synth-Papers dataset...")
    dataset = datasets.load_dataset("LeMaterial/LeMat-Synth-Papers", "full")
    print("Concatenating datasets...")
    all_data = concatenate_datasets(
        [dataset["chemrxiv"], dataset["omg24"], dataset["arxiv"]]
    )

    # Cast to large string to avoid truncation
    new_features = all_data.features.copy()
    new_features["text_paper"] = Value("large_string")
    all_data = all_data.cast(Features(new_features))

    # Filter to only papers in the keyword search results
    print("\nFiltering dataset to keyword-matched papers...")
    keyword_papers = all_data.filter(lambda x: x["id"] in keyword_ids)
    print(f"Filtered to {len(keyword_papers)} papers for LLM processing")

    # Apply LLM filtering
    print(f"\nProcessing {len(keyword_papers)} papers with LLM...")
    client = openai.OpenAI(base_url=VLLM_ENDPOINT, api_key="not-needed")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    results = []
    for paper in tqdm(keyword_papers, desc="LLM filtering"):
        results.append(
            process_example(
                paper,
                client,
                MODEL_NAME,
                tokenizer,
                selected_prompt=selected_prompt,
            )
        )

    # Add results to dataset
    keyword_papers = keyword_papers.add_column(
        "performance_vs_temperature_plot", results
    )

    # Filter to only papers with plots
    final_papers = keyword_papers.filter(
        lambda x: x["performance_vs_temperature_plot"]
    )

    print(f"\n{'=' * 60}")
    print("RESULTS:")
    print(f"{'=' * 60}")
    print(f"Papers processed: {len(keyword_papers)}")
    print(f"Papers with performance vs temperature plots: {len(final_papers)}")
    print(f"Success rate: {len(final_papers) / len(keyword_papers) * 100:.1f}%")
    print(f"{'=' * 60}\n")

    # Save to HuggingFace Hub
    print(f"Pushing dataset to {OUTPUT_REPO}...")
    final_papers.push_to_hub(OUTPUT_REPO)
    print("Dataset saved successfully!")

    # Download PDFs
    os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)
    sample_size = min(100, len(final_papers))
    print(f"\nDownloading {sample_size} sample PDFs...")

    for idx in tqdm(range(sample_size), desc="Downloading PDFs"):
        paper = final_papers[idx]
        pdf_url = paper["pdf_url"]
        filename = f"{paper['id']}_{os.path.basename(pdf_url.split('?')[0])}"
        filepath = os.path.join(DOWNLOAD_FOLDER, filename)

        try:
            response = requests.get(pdf_url, timeout=30)
            response.raise_for_status()
            with open(filepath, "wb") as f:
                f.write(response.content)
        except Exception as e:
            print(f"\nFailed to download {pdf_url}: {e}")

    print(f"\nDownloaded {sample_size} PDFs to {DOWNLOAD_FOLDER}")


if __name__ == "__main__":
    main()
