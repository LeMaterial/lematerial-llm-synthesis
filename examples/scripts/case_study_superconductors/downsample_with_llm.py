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

CRITICAL: If the ρ(T) or R(T) plot has multiple curves where the ONLY
difference is applied magnetic field (H, B, μ₀H) or pressure (GPa, kbar),
answer "no". Look carefully at figure captions and legends for values in
Tesla (T), Oe, kOe, GPa, or kbar.

Follow these two steps:
Step 1: Check if the paper has a ρ(T) or R(T) plot.
Step 2: If yes, check the figure caption/legend — do the different curves
represent different magnetic fields or pressures? If so, answer "no".

Answer with only yes or no.
If you are not sure, answer with no.

Paper: {paper_text}
Question: Does this paper contain a resistivity/resistance vs temperature plot
that is NOT just varying magnetic field or pressure?
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

CRITICAL — MAGNETIC FIELD EXCLUSION (most common false positive):
Many superconductor papers plot ρ(T) or R(T) at different applied magnetic \
fields to study the upper critical field Hc2. These plots show the \
superconducting transition broadening and shifting to lower temperatures \
with increasing field. The legend or caption will show values in Tesla (T), \
Oersted (Oe), or kOe. Common notations include:
  - H = 0, 1, 3, 5, 7, 9 T
  - μ₀H = 0, 0.1, 0.2 ... 0.8 T
  - H = 0T, 0.5T, 1T, 2T, 4T, 6T, 8T, 10T
  - H(θ = 0 deg), H = 15T
  - Applied fields of 0, 0.5, 1, 2 T
  - H_dc, H_a, H_ext, B_a in Tesla
  - "in magnetic fields" or "under various fields"
  - "resistivity curves for the x = 0.6 sample in magnetic fields"
  - ρ_xx(T) at different fields (longitudinal resistivity in field)
These are ALL excluded — answer "no".

Also EXCLUDE:
- ρ(T) or R(T) curves where the ONLY variation is applied pressure \
(GPa, kbar)
- Hall resistivity (ρ_xy or ρ_Hall) vs temperature
- Thermopower or Seebeck coefficient vs temperature
- Magnetoresistance (MR) vs field at fixed temperatures
- AC susceptibility (χ) vs temperature (this is not resistivity)
- Upper critical field Hc2(T) plots (these plot field vs temperature, \
not resistivity vs temperature)
- Only computational/theoretical curves with no experimental data

ANALYSIS STEPS — follow these before answering:
Step 1: Does the paper describe a ρ(T) or R(T) plot?
Step 2: Look at the figure captions and legends. What do the different \
curves represent? Are they labeled with field values (T, Oe, kOe), \
pressure values (GPa, kbar), or material/composition labels?
Step 3: If the curves are labeled with field or pressure values, answer "no". \
If they represent different materials, compositions, dopings, substitutions, \
or if it is a single-curve plot for one material, answer "yes".

Examples of VALID plots:
- "Resistivity ρ (mΩ cm) vs Temperature (K) for Nd[O₀.₈₉F₀.₁₁]FeAs" — \
single material showing superconducting transition
- "ρ(T) for BaFe₂As₂, Ba₀.₆K₀.₄Fe₂As₂, and Ba₀.₅K₀.₅OFe₂As₂" — \
comparing different compositions
- "Temperature dependence of resistivity for samples with x = 0, 0.05, 0.1" \
— comparing different dopings
- "R(T) for parent and Co-doped BaFe₂As₂" — comparing doping levels

Examples of INVALID plots:
- "ρ(T) at μ₀H = 0, 0.05, 0.1, 0.15 ... 0.8 T" — only varying magnetic field
- "R(T) under H = 0, 1, 3, 5, 7, 9 T" — only varying magnetic field
- "ρ(T) in applied fields of 0, 0.5, 1, 2, 5 T" — only varying magnetic field
- "resistivity curves for the x = 0.6 sample in magnetic fields" — \
only varying field for one composition
- "ρ_xx(T) at H = 15T, θ(deg) = 0, 40, 68, 73, 75, 77, 80, 85, 90" — \
varying field angle, still a field study
- "ρ(T) at 0T, 0.5T, 1T, 2T, 4T marked in legend" — field sweep
- "R(T) for CaLi₂ at 8, 11, 26, 36, 45 GPa" — only varying pressure
- "Hc2(T) determined from resistivity midpoint" — this is an Hc2 plot, \
not a ρ(T) plot

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
