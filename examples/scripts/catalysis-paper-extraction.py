#!/usr/bin/env python3
"""
Script to filter catalysis papers using a vLLM OpenAI-compatible server.
"""

import os
import re
import requests
from tqdm import tqdm
from transformers import AutoTokenizer
import datasets
from datasets import concatenate_datasets, Dataset
import openai
from tenacity import retry, stop_after_attempt, wait_exponential
from datasets import Features, Value

# --- Constants ---
MODEL_NAME = "mistralai/Ministral-3-14B-Instruct-2512"
MAX_MODEL_LEN = 50000
VLLM_ENDPOINT = "http://localhost:8000/v1"
OUTPUT_REPO = "amayuelas/LeMat-Synth-Papers-Catalysis-v2"
DOWNLOAD_FOLDER = "../data/catalysis_pdfs-v2"
KEYWORDS_CATALYSIS = ['catalysis', 'catalytic', 'catalyst', 'activation energy', 'TOF']
KEYWORDS_THERMAL_CATALYSIS = [
    'thermal catalysis', 'thermocatalytic', 'thermal catalytic', 'thermal catalyst', 'thermal treatment', 
    'thermocatalysis', "thermocatalysis", "thermo-catalysis", 'thermocatalyst', 'thermocatalytic'
]

# --- LLM Prompt ---
PROMPT = """You are provided with a scientific materials paper.
We want to know if the paper contains a plot of material performance as a function of temperature.
Read the paper carefully and determine if it contains a plot of material performance vs temperature.

Answer with only yes or no.
Do not include any other text in your answer.
If you are not sure, answer with no.

Start Example:
Paper: [paper_text]
Question: Does this paper contain a plot of material performance as a function of temperature?
Answer: [yes/no]
End Example.

Paper: {paper_text}
Question: Does this paper contain a plot of material performance as a function of temperature?
Answer:
"""

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def ask_llm_is_plot_of_material_performance_vs_temperature(text, client, model):
        message = PROMPT.format(paper_text=text)
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": message}],
                temperature=0,
                max_tokens=100,
            )
            answer = response.choices[0].message.content.strip().lower()
            return answer in ['yes', 'yes.'] or ('yes' in answer)
        except Exception as e:
            print(f"LLM call failed: {e}")
            return False

def process_example(example, client, model, tokenizer):
    text = example['text_paper']
    tokens = tokenizer.encode(text)
    if len(tokens) > MAX_MODEL_LEN:
        text = tokenizer.decode(tokens[:MAX_MODEL_LEN-150])
    return ask_llm_is_plot_of_material_performance_vs_temperature(text, client, model)

# --- Main Workflow ---
def main():
    # Load and filter dataset
    print("Loading dataset...")
    dataset = datasets.load_dataset("LeMaterial/LeMat-Synth-Papers", "full")
    print("Concatenating datasets...")
    all_data = concatenate_datasets([dataset['chemrxiv'], dataset['omg24'], dataset['arxiv']])
    # Cast to large string to avoid truncation
    new_features = all_data.features.copy()
    new_features["text_paper"] = Value("large_string")
    all_data = all_data.cast(Features(new_features))


    # Filter papers with catalysis keywords
    print("Filtering papers with catalysis keywords...")
    pattern = re.compile(r'\b(' + '|'.join(re.escape(k) for k in KEYWORDS_CATALYSIS) + r')\b', flags=re.IGNORECASE)
    def keyword_in_text(batch):
        texts = batch["text_paper"]
        return {"is_catalytic": [bool(pattern.search(t or "")) for t in texts]}
    catalysis_papers = all_data.map(
        keyword_in_text, 
        batched=True,
        remove_columns=all_data.column_names,  # IMPORTANT
        )
    all_data = all_data.add_column("is_catalytic", catalysis_papers["is_catalytic"])    
    print(f"Found {sum(all_data['is_catalytic'])} papers with catalysis keywords.")

    # Filter papers with thermal catalysis keywords
    print("Filtering papers with thermal catalysis keywords...")
    pattern = re.compile(r'\b(' + '|'.join(re.escape(k) for k in KEYWORDS_THERMAL_CATALYSIS) + r')\b', flags=re.IGNORECASE)
    def keyword_in_text(batch):
        texts = batch["text_paper"]
        return {"is_thermal_catalytic": [bool(pattern.search(t or "")) for t in texts]}
    thermal_catalysis_papers = all_data.map(
        keyword_in_text, 
        batched=True,
        remove_columns=all_data.column_names,  # IMPORTANT
        )
    all_data = all_data.add_column("is_thermal_catalytic", thermal_catalysis_papers["is_thermal_catalytic"])    
    print(f"Found {sum(all_data['is_thermal_catalytic'])} papers with thermal catalysis keywords.")

    # Filter papers containing the performance vs temperature plot
    thermal_catalysis_papers = all_data.filter(lambda x: x['is_catalytic'] and x['is_thermal_catalytic'])
    print(f"Found {len(thermal_catalysis_papers)} papers with catalysis and thermal catalysis keywords.")

    print("Processing papers with LLM...")
    client = openai.OpenAI(base_url=VLLM_ENDPOINT, api_key="not-needed")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    results = []
    for paper in tqdm(thermal_catalysis_papers, desc="Processing papers"):
        results.append(process_example(paper, client, MODEL_NAME, tokenizer))

    # Save results
    thermal_catalysis_papers = thermal_catalysis_papers.add_column("performance_vs_temperature_plot", results)
    print(f"Found {sum(thermal_catalysis_papers['performance_vs_temperature_plot'])} papers with performance vs temperature plot.")
    thermal_catalysis_papers.push_to_hub(OUTPUT_REPO)

    # Download PDFs
    plot_papers = thermal_catalysis_papers.filter(lambda x: x['performance_vs_temperature_plot'])
    os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)
    sample_size = min(100, len(plot_papers))
    print(f"Downloading {sample_size} PDFs...")
    for idx in range(sample_size):
        pdf_url = plot_papers[idx]['pdf_url']
        filename = os.path.basename(pdf_url.split("?")[0])
        filepath = os.path.join(DOWNLOAD_FOLDER, filename)
        try:
            response = requests.get(pdf_url, timeout=30)
            response.raise_for_status()
            with open(filepath, "wb") as f:
                f.write(response.content)
            print(f"Downloaded: {filename}")
        except Exception as e:
            print(f"Failed to download {pdf_url}: {e}")

if __name__ == "__main__":
    main()