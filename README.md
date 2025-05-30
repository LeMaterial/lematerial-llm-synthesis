![](assets/lematerial-logo.png)

# LeMaterial-Synthesis-Parser (LeMat-SynthP)
LeMaterial's LLM-based academic paper parsing module 

## Installation

This project uses **uv** as a package & project manager. See [uv’s README](https://github.com/astral-sh/uv?tab=readme-ov-file#installation) for installation instructions.

```bash
# 1. Clone & enter the repo
git clone https://github.com/LeMaterial/lematerial-llm-synthesis.git
cd lematerial-llm-synthesis

# 2. (First time only) create & seed venv
uv venv -p 3.11 --seed

# 3. Install dependencies & package
uv sync && uv pip install -e .
```

### macOS/Linux
```bash
cp .env.example .env
# Edit `.env` to add:
#   MISTRAL_API_KEY=your_api_key # if using Mistral models and Mistral OCR
#   OPENAI_API_KEY=your_api_key # if using OpenAI models
#   GEMINI_API_KEY=your_api_key # if using Gemini models
```


### Windows

* Search bar --> Edit the system environment variables --> Advanced --> click "Environment Variables..."
* Under "User variables for <your-username>" click "New" and add each:
    * Variable name: MISTRAL_API_KEY; Value: your_api_key
    * Variable name: OPENAI_API_KEY; Value: your_api_key
    * Variable name: GEMINI_API_KEY; Value: your_api_key

For any platform you can always load .env-style keys in code via `os.environ.get(...)`.

### Verify installation

```
uv run python -c "import llm_synthesis"
```

No errors? You're all set!

## Usage

### Text Extraction

For usage in a notebook, cf. `notebooks/pdf_extraction.ipynb`

```sh
uv run scripts/extract_text_from_pdfs.py --base-path <local or gcs path to the working folder> --process <"docling" or "mistral">
```

For example, this will extract text from `./data/pdf_papers` and write the result to `./data/txt_papers/docling` using Docling:

```sh
uv run scripts/extract_text_from_pdfs.py --base-path data/ --process docling
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
