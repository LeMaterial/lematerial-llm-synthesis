# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LeMaterial-Synthesis (`llm_synthesis`) is a multi-modal toolbox for extracting structured synthesis procedures and performance data from materials science literature at scale. It uses LLMs (via DSPy + LiteLLM) to parse scientific papers into structured Pydantic ontologies.

## Development Setup

```bash
uv venv -p 3.11 --seed
uv sync && uv pip install -e .
uvx pre-commit install
```

Verify: `uv run python -c "import llm_synthesis"`

API keys are loaded from a `.env` file (`source .env`). Supported: `MISTRAL_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`, `ANTHROPIC_API_KEY`, plus `OPENROUTER_*` keys for models routed through OpenRouter.

## Commands

```bash
# Lint and format
uvx ruff check                # lint
uvx ruff check --fix          # lint with autofix
uvx ruff format               # format

# Run the main extraction pipeline (uses Hydra config)
uv run examples/scripts/deployment/extract_synthesis_procedure_from_text.py \
  data_loader=default synthesis_extraction=default material_extraction=default \
  judge=default result_save=default

# Run with local data
uv run examples/scripts/deployment/extract_synthesis_procedure_from_text.py \
  data_loader=local data_loader.architecture.data_dir="/path/to/markdown"

# Add a dependency
uv add <package-name>         # adds to pyproject.toml
uv pip install <package-name> # installs without adding to pyproject.toml
```

## Linting

Ruff is configured in `pyproject.toml`: target Python 3.11, line length 80, rules `E/F/I/N/UP/RUF` (ignoring `RUF001-003`). `__init__.py` files ignore `F401`. Notebooks are excluded from linting. Pre-commit runs ruff lint+format and `nbstripout` on every commit.

## Architecture

### Core Pipeline Flow

The system processes scientific papers through a multi-step pipeline:
1. **Data Loading** → Load papers from HuggingFace or local filesystem
2. **Material Extraction** → Identify synthesized materials in paper text
3. **Synthesis Extraction** → Extract structured synthesis procedures per material
4. **Judge Evaluation** → Score extraction quality (optional)
5. **Figure/Plot Extraction** → Extract quantitative data from figures
6. **Performance Linking** → Link plot series data back to specific materials

The main orchestrator is `SynthesisPerformancePipeline` in `src/llm_synthesis/services/pipelines/synthesis_performance_pipeline.py`. It supports both sync (`process_paper`) and async (`process_paper_async`) execution with semaphore-based concurrency control.

### Key Abstractions (`src/llm_synthesis/`)

- **`transformers/base.py`**: `ExtractorInterface[T, R]` — base class for all extractors. Extends `dspy.Module`, requires a `forward(input: T) -> R` method. Concrete extractors live in subdirectories: `material_extraction/`, `synthesis_extraction/`, `figure_extraction/`, `plot_extraction/`, `performance_linking/`.
- **`metrics/base.py`**: `MetricInterface[T]` — base class for evaluation metrics. The `judge/` subpackage contains LLM-as-judge evaluators (`GeneralSynthesisJudge`, `LinkingJudge`).
- **`models/ontologies/general.py`**: `GeneralSynthesisOntology` — the central Pydantic schema for structured synthesis data (target compound, synthesis method, materials, process steps with conditions/equipment). This is the output type of synthesis extraction.
- **`models/paper.py`**: `Paper` and `PaperWithSynthesisOntologies` — data models for papers flowing through the pipeline.
- **`data_loader/paper_loader/`**: `PaperLoaderInterface` with implementations for HuggingFace datasets (`hf_paper_loader.py`) and local filesystems (`fs_paper_loader.py`).
- **`services/storage/`**: File storage abstraction (local and GCS) for persisting results.
- **`utils/llms.py`**: `LLM_REGISTRY` — central registry of all supported LLM configurations. `SystemPrefixedLM` wraps `dspy.LM` to inject system prompts and track costs.

### Configuration

The project uses **Hydra** for configuration. Config files live in `examples/config/` with groups for `data_loader`, `synthesis_extraction`, `material_extraction`, `judge`, `result_save`, and `plot_extraction`. Override any config value from the command line using Hydra's dot notation.

System prompts for LLM calls are stored as text files in `examples/system_prompts/`.

### LLM Integration

All LLM calls go through **DSPy** with **LiteLLM** as the backend. Models are registered in `LLM_REGISTRY` (`utils/llms.py`) and support Gemini, Claude, GPT, Mistral, and OpenRouter-proxied models (Qwen, Kimi, DeepSeek). The multi-LLM pipeline runs multiple models and uses agreement-based scoring.

### Commit Style

Uses Conventional Commits: `feat:`, `fix:`, `docs:`, `style:`, `refactor:`, `test:` with optional scope, e.g. `feat(extraction): support image embedding in markdown output`.

## Key Dependencies

- `dspy` — LLM programming framework (structured extraction via signatures)
- `litellm` — unified LLM API gateway (pinned `<1.82`)
- `hydra-core` — configuration management
- `pydantic` — data models and ontology schemas
- `anthropic` — direct Claude API calls (used for plot data extraction)
- `docling` — document parsing
- `pymupdf` — PDF processing
