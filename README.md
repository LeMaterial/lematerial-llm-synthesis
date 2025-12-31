![](assets/lematerial-logo.png)

# LeMaterial-Synthesis

An open-source multi-modal toolbox for extracting structured synthesis procedures and performance data from materials science literature at scale. This repository contains the implementations of [LeMat-Synth v1.0](https://arxiv.org/abs/2510.26824) (published on the arXiv and presented at NeurIPS AI4Mat 2025) plus the extendable codebase for usecases in materials science.

![](assets/overview.png)

---

## Quick Start

<details>
<summary><b>Installation Instructions</b></summary>

### Prerequisites

This project uses **uv** as a package & project manager. See [uv's README](https://github.com/astral-sh/uv?tab=readme-ov-file#installation) for installation instructions.

### Setup
```bash
# 1. Clone & enter the repo
git clone https://github.com/LeMaterial/lematerial-llm-synthesis.git
cd lematerial-llm-synthesis

# 2. (First time only) create & seed venv
uv venv -p 3.11 --seed

# 3. Install dependencies & package
uv sync && uv pip install -e .
```

### API Key Configuration

<details>
<summary><b>macOS/Linux</b></summary>
```bash
cp .env.example .env
# Edit `.env` to add:
#   MISTRAL_API_KEY=your_api_key # if using Mistral models and Mistral OCR
#   OPENAI_API_KEY=your_api_key # if using OpenAI models
#   GEMINI_API_KEY=your_api_key # if using Gemini models
#   ANTHROPIC_API_KEY=your_api_key # if using Anthropic models (Claude, image extraction)
```

Before running the scripts, you need to load your API keys. For this you need to source the .env file. Run:
```bash
source .env
```

</details>

<details>
<summary><b>Windows</b></summary>

- Search bar → Edit the system environment variables → Advanced → click "Environment Variables..."
- Under "User variables for <your-username>" click "New" and add each:
  - Variable name: `MISTRAL_API_KEY`; Value: `your_api_key`
  - Variable name: `OPENAI_API_KEY`; Value: `your_api_key`
  - Variable name: `GEMINI_API_KEY`; Value: `your_api_key`
  - Variable name: `GOOGLE_APPLICATION_CREDENTIALS`; Value: `C:\path\to\service-account.json`

</details>

**Note:** For any platform you can always load .env-style keys in code via `os.environ.get(...)`.

### Verify Installation
```bash
uv run python -c "import llm_synthesis"
```

No errors? You're all set!

</details>

---

## Dataset Access

<details>
<summary><b>Fetching HuggingFace Dataset LeMat-Synth</b></summary>

The data is hosted as a LeMaterial Dataset on HuggingFace: [LeMat-Synth](https://huggingface.co/datasets/LeMaterial/LeMat-Synth/)

### Access Steps

1. **Apply for access** (request will be instantly approved)
2. **Install HuggingFace CLI** ([guide](https://huggingface.co/docs/huggingface_hub/en/guides/cli))
   - Recommended: `pip install -U "huggingface_hub[cli]"`
   - Or (macOS): `brew install huggingface-cli`
3. **Login with access token**: `huggingface-cli login`

### Available Datasets

- **[LeMat-Synth](https://huggingface.co/datasets/LeMaterial/LeMat-Synth/)**: Synthesis procedures and images in structured (per-synthesis) format
- **[LeMat-Synth-Papers](https://huggingface.co/datasets/LeMaterial/LeMat-Synth-Papers/)**: Intermediate dataset storing papers in per-paper format

</details>

---

## Usage

### Extract from HuggingFace Dataset
```bash
uv run examples/scripts/extract_synthesis_procedure_from_text.py \
  data_loader=default \
  synthesis_extraction=default \
  material_extraction=default \
  judge=default \
  result_save=default
```

### Extract Synthesis Locally
```bash
uv run examples/scripts/extract_synthesis_procedure_from_text.py \
  data_loader=local \
  data_loader.architecture.data_dir="/path/to/markdown" \
  synthesis_extraction=default \
  material_extraction=default \
  judge=default \
  result_save=default
```

### Extract Images Locally

*Work in Progress*

### Customize LeMat-Synth
*Work in Progress*

### Thermocatalysis Case Study

*Work in Progress*

Filter down:
```bash
uv run examples/scripts/case_study_thermocatalysis/keyword_search.py
uv run examples/scripts/case_study_thermocatalysis/downsample_with_llm.py --prompt default
uv run examples/scripts/case_study_thermocatalysis/downsample_with_llm.py --prompt long
```

---