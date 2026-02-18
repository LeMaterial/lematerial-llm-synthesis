"""
Script to apply LLM filtering to superconductor papers
identified by keyword search.
Loads paper IDs from pkl file and applies LLM inference to check for
resistivity vs temperature plots.

We want papers that show ρ (or R) vs T plots — either a single curve for one
material or multiple curves comparing different
compositions/dopings/substitutions.
We exclude papers where the only variation between curves is magnetic field or
pressure.
"""

import argparse
import os
import pickle

import datasets
import requests
from datasets import Features, Value, concatenate_datasets
from google import genai
from google.genai import types
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

# --- Constants ---
MODEL_NAME = "gemini-2.5-flash"
DOWNLOAD_FOLDER = "../data/superconductor_pdfs-filtered-v1"
PKL_FILE = "results/db_superconductors.pkl"

# --- LLM Prompt (short) ---
PROMPT = """You are provided with a scientific materials paper about
superconductors. We want to know if the paper contains a plot of
electrical resistivity (ρ) or resistance (R) as a function of temperature (T).

The plot can show a single curve for one material, or multiple curves comparing
different materials, compositions, dopings, or substitutions.

Important: The y-axis is typically labeled as ρ (with units like mΩ·cm, μΩ·cm,
Ω·cm) or R (with units like Ω, mΩ). It may sometimes say "resistivity" but
do not rely on that word alone — most papers use the symbol ρ or R instead.

We do NOT want plots where the only variation between curves is:
- Applied magnetic field (H or B)
- Applied pressure (GPa, kbar)
Those are excluded.

Answer with only yes or no.
Do not include any other text in your answer.
If you are not sure, answer with no.

Start Example:
Paper: [paper_text]
Question: Does this paper contain a resistivity/resistance vs temperature plot?
Answer: [yes/no]
End Example.

Paper: {paper_text}
Question: Does this paper contain a resistivity/resistance vs temperature plot?
Answer:
"""

# --- LLM Prompt (long/detailed) ---
PROMPT_LONG = """You are a scientific paper analyzer specializing in \
superconductor research.

Your task: Determine if this paper contains QUANTITATIVE line/curve plots \
showing electrical resistivity or resistance as a function of temperature.

The plot can show a SINGLE curve for one material (e.g., ρ(T) for \
Nd[O₀.₈₉F₀.₁₁]FeAs showing a superconducting transition), OR multiple \
curves comparing different materials, compositions, dopings, or substitutions.

IMPORTANT AXIS CONVENTIONS:
- The y-axis may be labeled in several ways — do not rely on the word \
"resistivity" alone, but DO accept it if present. Common labels include:
  - ρ (rho) with units: mΩ·cm, μΩ·cm, Ω·cm, mΩ*cm, μΩ*cm
  - R with units: Ω, mΩ, kΩ
  - ρ/ρ₃₀₀ or R/R₃₀₀ (normalized resistivity/resistance)
  - ρ(T) or R(T)
  - "Resistivity" or "Resistance" (less common but valid)
- The x-axis should show TEMPERATURE:
  - T with units: K (Kelvin), °C, or °F
  - Temperature (K)

REQUIRED CRITERIA (all must be met):
1. The plot must show TEMPERATURE on the x-axis (in K, °C, or similar)
2. The plot must show RESISTIVITY (ρ) or RESISTANCE (R) on the y-axis \
(see axis conventions above)
3. The plot must be a LINE CHART or CURVE \
(showing trends across multiple temperatures)
4. The plot must show EXPERIMENTAL data

EXCLUDE papers that ONLY have:
- ρ(T) or R(T) curves where the ONLY variation is applied magnetic field \
(H or B) — these show field-induced broadening of the superconducting \
transition and are NOT what we want
- ρ(T) or R(T) curves where the ONLY variation is applied pressure \
(GPa, kbar) — we do NOT want pressure-dependent studies
- Hall resistivity (ρ_xy or ρ_Hall) vs temperature
- Thermopower or Seebeck coefficient vs temperature
- Magnetoresistance (MR) vs field at fixed temperatures
- AC susceptibility (χ) vs temperature (this is not resistivity)
- Only computational/theoretical curves with no experimental data

Examples of VALID plots:
- "Resistivity ρ (mΩ cm) vs Temperature (K) for Nd[O₀.₈₉F₀.₁₁]FeAs" — \
single material showing superconducting transition
- "ρ(T) for BaFe₂As₂, Ba₀.₆K₀.₄Fe₂As₂, and Ba₀.₅K₀.₅OFe₂As₂" — \
comparing different compositions
- "Temperature dependence of resistivity for samples with x = 0, 0.05, 0.1" \
— comparing different dopings

Examples of INVALID plots:
- "ρ(T) at μ₀H = 0, 0.05, 0.1, 0.15 ... 0.8 T" — only varying magnetic field
- "R(T) under H = 0, 1, 3, 5, 7, 9 T" — only varying magnetic field
- "R(T) for CaLi₂ at 8, 11, 26, 36, 45 GPa" — only varying pressure

Answer ONLY with "yes" or "no". No other text.
If uncertain, answer "no".

Paper: {paper_text}

Does this paper contain a resistivity/resistance vs temperature plot \
(NOT just varying magnetic fields or pressures)?
Answer:"""


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
)
def ask_llm_has_resistivity_vs_temperature_plot(text, client, selected_prompt):
    message = selected_prompt.format(paper_text=text)
    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=message,
            config=types.GenerateContentConfig(
                temperature=0, max_output_tokens=100
            ),
        )
        answer = response.text.strip().lower()
        return answer in ["yes", "yes."] or ("yes" in answer)
    except Exception as e:
        print(f"LLM call failed: {e}")
        return False


def process_example(example, client, selected_prompt):
    text = example["text_paper"]
    return ask_llm_has_resistivity_vs_temperature_plot(
        text, client, selected_prompt
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
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get(
        "GOOGLE_API_KEY"
    )
    if not api_key:
        raise OSError(
            "Set GEMINI_API_KEY (or GOOGLE_API_KEY) env var. "
            "Use 'export GEMINI_API_KEY=...' so subprocesses inherit it."
        )
    client = genai.Client(api_key=api_key)

    results = []
    for paper in tqdm(keyword_papers, desc="LLM filtering"):
        results.append(
            process_example(paper, client, selected_prompt=selected_prompt)
        )

    # Add results to dataset
    keyword_papers = keyword_papers.add_column(
        "resistivity_vs_temperature_plot", results
    )

    # Filter to only papers with plots
    final_papers = keyword_papers.filter(
        lambda x: x["resistivity_vs_temperature_plot"]
    )

    print(f"\n{'=' * 60}")
    print("RESULTS:")
    print(f"{'=' * 60}")
    print(f"Papers processed: {len(keyword_papers)}")
    print(f"Papers with resistivity vs temperature plots: {len(final_papers)}")
    print(f"Success rate: {len(final_papers) / len(keyword_papers) * 100:.1f}%")
    print(f"{'=' * 60}\n")

    # Save to HuggingFace Hub
    print("Pushing dataset ")
    final_papers.push_to_hub(
        "LeMaterial/LeMat-Synth-Papers",
        config_name="superconductor_keywords_and_LLM",
        split="full",
        create_pr=True,
        token=True,
    )
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
