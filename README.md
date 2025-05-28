![](assets/lematerial-logo.png)

# LeMaterial-Synthesis-Parser (LeMat-SynthP)
LeMaterial's LLM-based academic paper parsing module 

## Usage

### Installation
This project uses uv as a package and project manager, see [https://github.com/astral-sh/uv?tab=readme-ov-file#installation](here) for installation instructions.

```bash
git clone https://github.com/LeMaterial/lematerial-synthesis-parser
# 1️⃣ only the first time to start the venv
uv venv -p 3.11 --seed
# Install project dependencies using uv
uv sync 
# Install the package in editable mode
uv pip install -e .
```

If using Mistral OCR, make sure to save your MISTRAL_API_KEY in the `.env` file:
```bash
MISTRAL_API_KEY=your_api_key_here
```

If using Gemini or OpenAI LLMs, make sure to add `GEMINI_API_KEY` and `OPENAI_API_KEY` to your `.env` file. To access Google Cloud storage, make sure to include the path to your service account under `GOOGLE_APPLICATION_CREDENTIALS=/path/to/service/account.json`


## Using the command line interface

### Text Extraction

For usage in a notebook, cf. `notebooks/pdf_extraction.ipynb`

```sh
uv run scripts/extract_text_from_pdfs.py --base-path <local or gcs path to the working folder> --process <"docling" or "mistral">
```

For example, this will extract text from `gcs://entalpic-prod-llm-synthesis-papers/test/pdf_files` and write the result to `gcs://entalpic-prod-llm-synthesis-papers/test/txt_files/docling` using Docling:

```sh
uv run scripts/extract_text_from_pdfs.py --base-path gcs://entalpic-prod-llm-synthesis-papers/test --process docling
```

For local usage, this will extract text from `data/pdf_files` and write the result to `data/txt_files/docling` using Docling:

```sh
uv run scripts/extract_text_from_pdfs.py --base-path data --process docling
```

### Extracting a synthesis procedure from the parsed text

For usage in a notebook, cf. `notebooks/synthesis_procedure_extraction.ipynb`

**Benchmark – Sweeping over different configurations**. DSPy designs LLM pipelines in a very modular way: The quality of the output is influenced by the LLM, prompting strategy, pre-processing steps etc.
In order to keep track of every *moving part* of our model, we use [hydra](https://hydra.cc/) to track experiments. We can run a specific configuration directly from the command line:

```
uv run scripts/extract_synthesis_procedure_from_text.py synthesis_extraction.lm.name=gpt-4o-mini
```

To sweep over several configurations, use the flag `--multirun`:

```
uv run scripts/extract_ontology_from_text.py --multirun \
    synthesis_extraction.lm.name=gemini-2.0-flash,gemini-2.5-flash,gpt-4o
```

The results of the runs are saved in `results/single_run` and `results/multi_run`, respectively. The synthesis paragraphs and structured synthesis procedures are saved and can be inspected there.
**Metrics**: Note that the metrics are an arbitrary, random number at this stage.
