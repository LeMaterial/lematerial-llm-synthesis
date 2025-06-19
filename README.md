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

#### Load your API keys

Before running the scripts, you need to load your API keys. For this you need to source the .env file. Run:

```bash
source .env
```

### Windows

- Search bar --> Edit the system environment variables --> Advanced --> click "Environment Variables..."
- Under "User variables for <your-username>" click "New" and add each:
  - Variable name: MISTRAL_API_KEY; Value: your_api_key
  - Variable name: OPENAI_API_KEY; Value: your_api_key
  - Variable name: GEMINI_API_KEY; Value: your_api_key
  - Variable name: GOOGLE_APPLICATION_CREDENTIALS; Value: C:\path\to\service-account.json

For any platform you can always load .env-style keys in code via `os.environ.get(...)`.

### Verify installation

```
uv run python -c "import llm_synthesis"
```

No errors? You're all set!

## Fetching Huggingface Dataset LeMat-Synth

The data is hosted as a LeMaterial Dataset on HuggingFace ([see here](https://huggingface.co/datasets/LeMaterial/LeMat-Synth/settings)).
In order to download and use it, apply for access once (the request will be instantly approved). Install the huggingafce-cli (use [this guide](https://huggingface.co/docs/huggingface_hub/en/guides/cli), recommended: `pip install -U "huggingface_hub[cli]"` or `brew install huggingface-cli` (macOS)) and log in with an access token (`huggingface-cli login`).

## Usage

### Text Extraction

For usage in a notebook, cf. `notebooks/pdf_extraction.ipynb`

```sh
uv run examples/scripts/extract_text_from_pdfs.py --input-path <local folder containing the pdfs> --output-path <local folder where the extracted text will be saved> --process <"docling" or "mistral">
```

For example, this will extract text from `./data/pdf_papers` and write the result to `./data/txt_papers/docling` using Docling:

```sh
uv run examples/scripts/extract_text_from_pdfs.py --input-path data/pdf_papers --output-path data/txt_papers/docling --process docling
```

### Extracting a synthesis procedure from the parsed text

For usage in a notebook, cf. `notebooks/synthesis_procedure_extraction.ipynb`

**Benchmark – Sweeping over different configurations**. DSPy designs LLM pipelines in a very modular way: The quality of the output is influenced by the LLM, prompting strategy, pre-processing steps etc.
In order to keep track of every _moving part_ of our model, we use [hydra](https://hydra.cc/) to track experiments. We can run a specific configuration directly from the command line:

```
uv run examples/scripts/extract_synthesis_procedure_from_text.py synthesis_extraction.architecture.lm.llm_name=gpt-4o-mini
```

To sweep over several configurations, use the flag `--multirun`:

```
uv run examples/scripts/extract_synthesis_procedure_from_text.py --multirun \
    synthesis_extraction.architecture.lm.llm_name=gemini-2.0-flash,gemini-2.5-flash,gpt-4o
```

The results of the runs are saved in `results/single_run` and `results/multi_run`, respectively. The synthesis paragraphs and structured synthesis procedures are saved and can be inspected there.
**Metrics**: Note that the metrics are an arbitrary, random number at this stage.
