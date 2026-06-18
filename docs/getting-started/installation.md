# Installation

## Prerequisites

- **Python 3.11+**
- **[uv](https://github.com/astral-sh/uv)** — the package manager used by this project

Install `uv` (one-time):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Install the package

```bash
# 1. Clone the repository
git clone https://github.com/LeMaterial/lematerial-llm-synthesis.git
cd lematerial-llm-synthesis

# 2. Create a virtual environment and install all dependencies
uv sync

# 3. Install the llm_synthesis package in editable mode
uv pip install -e .

# 4. Verify the installation
uv run python -c "import llm_synthesis; print('OK')"
```

## Set up API keys

Copy the example environment file and add your keys:

```bash
cp .env.example .env
```

Edit `.env` and fill in the keys you need:

```
GEMINI_API_KEY=...        # Required for synthesis extraction (default model)
ANTHROPIC_API_KEY=...     # Required for performance plot extraction
MISTRAL_API_KEY=...       # Required for Mistral OCR (PDF extraction)
OPENAI_API_KEY=...        # Optional: OpenAI models
OPENROUTER_QWEN_API_KEY=... # Optional: Qwen models
OPENROUTER_DEEPSEEK_API_KEY=... # Optional: DeepSeek models
```

You only need the keys for the models you actually plan to use.
A free **Gemini API key** from [aistudio.google.com](https://aistudio.google.com/app/apikey)
is sufficient for running the default extraction pipeline.

## Optional: PDF support via Playwright

If you want to download PDFs from journal websites, install the Playwright browsers:

```bash
uv run playwright install
```

## Optional: install pre-commit hooks (for contributors)

```bash
uvx pre-commit install
```

## Verify

```bash
lemat-synth --help
```

You should see the `extract` and `batch` subcommands listed.
