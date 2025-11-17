# %%
# imports
import os
import requests
import re
import pandas as pd 
from tqdm import tqdm

from transformers import AutoTokenizer
import datasets
from datasets import concatenate_datasets, Dataset
import openai


# %% [markdown]
# ## Download Papers

# %%
dataset = datasets.load_dataset(
    "LeMaterial/LeMat-Synth-Papers",
    "full"
    )

# %%
print(dataset.column_names)

# %%
concated_data = concatenate_datasets([dataset['chemrxiv'],dataset['omg24'], dataset['arxiv']])

# %%
len(concated_data)

# %% [markdown]
# ## Filter data

# %%
df = concated_data.to_pandas()
df = df.drop_duplicates(subset='doi', keep='first')
all_data = Dataset.from_pandas(df)
all_data = all_data.remove_columns(['__index_level_0__'])

# %%
len(all_data)

# %%
# Define keywords (all lowercase)
keywords = ['catalysis', 'catalytic', 'catalyst', 'activation energy', 'TOF']

# Compile regular expression for fast search
pattern = re.compile(r'\b(' + '|'.join(re.escape(k) for k in keywords) + r')\b', flags=re.IGNORECASE)

def keyword_in_abstract(example):
    text = example.get('text_paper', '') or ''
    return bool(pattern.search(text))

catalysis_papers = all_data.filter(keyword_in_abstract)

# %%
len(catalysis_papers)

# %% [markdown]
# ## LLM Filtering

# %%
prompt = """You are provided with a paper. We want to know if the paper uses a catalytic process to synthesize the material. Answer with only yes or no. 
Start Example: 
Paper: [paper_text]
Question: Does this paper use a catalytic process to synthesize the material?
Answer: [yes/no]
End Example.


Paper: {paper_text}
Question: Does this paper use a catalytic process to synthesize the material?
Answer: 
"""

# %%
# Assume you have an OpenAI-compatible endpoint, e.g.:
def ask_llm_is_catalytic(text, client, model):
    """
    Query an LLM to determine if a paper uses a catalytic process to synthesize the material.
    Returns True if catalytic, False if not, None if unclear.
    Uses the provided client to make the API call.
    """
    message = prompt.format(paper_text=text)
    try:
        # Use the provided client to make the LLM API request (OpenAI-compatible)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": message}
            ],
            temperature=0,
            max_tokens=100,
        )
        # Try to extract the answer in the same way as before
        answer = response.choices[0].message.content.strip().lower()
        if answer in ['yes', 'yes.']:
            return True
        elif answer in ['no', 'no.']:
            return False
        elif 'yes' in answer:
            return True
        elif 'no' in answer:
            return False
        else:
            return False
    except Exception as e:
        print(f"LLM call failed: {e}")
        return False

# %%
# create openai client
client = openai.OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
model_name = "mistralai/Mistral-Small-24B-Instruct-2501"
max_model_len = 32768
tokenizer = AutoTokenizer.from_pretrained(model_name)

# %%
# Example usage (evaluate for first 5 papers)
# results = [ask_llm_is_catalytic(p['text_paper']) for p in catalysis_papers.select(range(5))]

from concurrent.futures import ThreadPoolExecutor, as_completed

def preprocess_text(example):
    # tokenize the text, check if it's more than max_model_len, if so, split it into chunks, take the first chunk and pass it to the LLM
    text = example['text_paper']
    tokens = tokenizer.encode(text)
    if len(tokens) > max_model_len:
        text = tokenizer.decode(tokens[:max_model_len-150])
    return text

def process_example(example):
    text = preprocess_text(example)
    return ask_llm_is_catalytic(text, client, model_name)

results = []
with ThreadPoolExecutor(max_workers=8) as executor:
    futures = {executor.submit(process_example, example): i for i, example in enumerate(catalysis_papers)}
    for f in tqdm(as_completed(futures), total=len(catalysis_papers), desc="LLM Catalysis Check"):
        results.append(f.result())


# %% [markdown]
# ## Store results

# %%
## Create a new column with the results
catalysis_papers_llm = catalysis_papers.add_column("is_catalytic", results)
# select the columns where is_catalytic == True
catalysis_papers_llm_filtered = catalysis_papers_llm.filter(lambda x: x['is_catalytic'] == True)
catalysis_papers_llm_filtered.push_to_hub("amayuelas/LeMat-Synth-Papers-Catalysis")

# %%
## Download and Save some papers in folder
# Download a sample of PDFs from the filtered dataset
download_folder = "../data/catalysis_pdfs"
os.makedirs(download_folder, exist_ok=True)

sample_size = min(100, len(catalysis_papers_llm_filtered))  # Download 100 or fewer if less available

for idx in range(sample_size):
    pdf_url = catalysis_papers_llm_filtered[idx]['pdf_url']
    filename = os.path.basename(pdf_url.split("?")[0])
    filepath = os.path.join(download_folder, filename)
    try:
        response = requests.get(pdf_url, timeout=30)
        response.raise_for_status()
        with open(filepath, "wb") as f:
            f.write(response.content)
        print(f"Downloaded: {filename}")
    except Exception as e:
        print(f"Failed to download {pdf_url}: {e}")


# %%



