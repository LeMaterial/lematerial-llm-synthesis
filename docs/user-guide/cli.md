# CLI Reference

The `lemat-synth` command-line tool lets you extract structured synthesis
procedures from materials science papers without writing any Python code.

```bash
lemat-synth extract <paper>   [key=value ...]
lemat-synth batch   <folder>  [key=value ...]
```

All settings — models, prompts, domain, output path — live in
`config/cli.yaml` at the repository root.  You can override any of them
directly on the command line using [Hydra](https://hydra.cc) `key=value`
syntax.

## Quick Reference: All Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| **Models & API** |
| `synthesis_model` | string | `gemini/gemini-2.0-flash` | Main extraction model (LiteLLM format) |
| `material_model` | string | `gemini/gemini-2.5-flash-lite` | Fast model for material-list extraction |
| `judge_model` | string | *(mirrors `synthesis_model`)* | Quality evaluation model |
| `linker_model` | string | `gemini/gemini-3-pro-preview` | Links plots to materials (requires `with_performance=true`) |
| `plot_model` | string | `claude-sonnet-4-20250514` | Claude model for plot data extraction (requires `with_performance=true`) |
| `api_base` | string | `null` | Custom API base URL, e.g. `https://openrouter.ai/api/v1` |
| `synthesis_api_key_env` | string | `null` | Env var name holding synthesis model API key |
| `material_api_key_env` | string | `null` | Env var name holding material model API key |
| `judge_api_key_env` | string | *(mirrors `synthesis_api_key_env`)* | Env var name for judge model |
| `linker_api_key_env` | string | `null` | Env var name for linker model |
| **Pipeline Behavior** |
| `domain` | choice | `generic` | Plot filtering: `generic`, `catalysis`, `superconductors`, `electrochemistry` |
| `with_performance` | bool | `false` | Extract performance data and link to materials (requires Claude API key) |
| `output_dir` | path | `results` | Output directory for results |
| `pdf_extractor` | choice | `docling` | PDF extraction backend: `docling` (local) or `mistral` (API-based) |
| `figure_segmenter` | choice | `dino` | Figure segmentation: `dino` or `florence` |
| `florence_repo_id` | string | `amayuelas/plot-visualization-florence-2-lora-32` | HuggingFace LoRA adapter ID (when `figure_segmenter=florence`) |
| **Batch Only** |
| `max_papers` | int | `null` | Maximum papers to process (`null` = all) |
| `skip_existing` | bool | `true` | Skip papers already in output directory |
| `max_papers_parallel` | int | `4` | Concurrent papers to process |
| **Prompts** |
| `prompts.synthesis_system` | string | *(see below)* | System message for synthesis extraction |
| `prompts.synthesis_instructions` | string | *(see below)* | Task instructions for synthesis extractor |
| `prompts.material_instructions` | string | *(see below)* | Task instructions for material extractor |
| Other prompt keys | string | *(see below)* | See [Customising prompts](#customising-prompts) |

---

## Installation & setup

Install the package (one-time):

```bash
uv pip install -e .
```

Create a `.env` file in the repository root with the API keys you need:

```dotenv
GEMINI_API_KEY=...                  # Google Gemini models (synthesis default)
ANTHROPIC_API_KEY=...               # Claude models and with_performance=true
OPENAI_API_KEY=...                  # OpenAI GPT models
MISTRAL_API_KEY=...                 # Mistral models or pdf_extractor=mistral
# OpenRouter — one key slot per model family (use synthesis_api_key_env etc.)
OPENROUTER_QWEN_API_KEY=...         # qwen3.5-35b-a3b
OPENROUTER_KIMI_API_KEY=...         # moonshotai/kimi-k2.5
OPENROUTER_DEEPSEEK_API_KEY=...     # deepseek/deepseek-v3.2
```

The CLI loads `.env` automatically on every run.

---

## `lemat-synth extract` — single paper

```
lemat-synth extract INPUT_FILE [key=value ...]
```

`INPUT_FILE` can be a plain-text file (`.txt`), a Markdown file (`.md`
produced by a PDF extractor), or a PDF (`.pdf` — Docling is used
automatically).

### Examples

#### Basic usage

```bash
# Uses all defaults from config/cli.yaml
lemat-synth extract paper.txt

# Custom output folder
lemat-synth extract paper.txt output_dir=my_results/
```

#### Common customizations

```bash
# Use a different synthesis model
lemat-synth extract paper.txt synthesis_model=anthropic/claude-sonnet-4-6

# Domain-specific plot filtering (catalysis, superconductors, or electrochemistry)
lemat-synth extract paper.txt domain=catalysis

# Extract performance data and link plots to materials (requires ANTHROPIC_API_KEY)
lemat-synth extract paper.txt with_performance=true

# Use Mistral OCR for better PDF extraction (requires MISTRAL_API_KEY)
lemat-synth extract paper.pdf pdf_extractor=mistral

# Override the synthesis extraction prompt
lemat-synth extract paper.txt \
    "prompts.synthesis_instructions=Extract only the primary synthesis route, ignoring alternative procedures."
```

#### Advanced: OpenRouter with multiple API keys

Route different models through different OpenRouter slots to manage rate limits or costs:

```bash
# All models through OpenRouter (Gemini Flash for synthesis, Claude for performance)
lemat-synth extract data/cipollone_2022.pdf \
    api_base="https://openrouter.ai/api/v1" \
    pdf_extractor=mistral \
    material_model="openrouter/google/gemini-3.1-pro-preview" \
    material_api_key_env=GEMINI_API_KEY \
    synthesis_model="openrouter/google/gemini-3-flash-preview" \
    synthesis_api_key_env=GEMINI_API_KEY \
    linker_model="openrouter/google/gemini-3.1-pro-preview" \
    linker_api_key_env=GEMINI_API_KEY \
    output_dir="results/"
```

#### Advanced: Extract with performance linking (OpenRouter)

Extract synthesis procedures and link extracted plot data to synthesized materials:

```bash
# Same as above, plus performance extraction using Claude via OpenRouter
lemat-synth extract data/cipollone_2022.pdf \
    api_base="https://openrouter.ai/api/v1" \
    material_model="openrouter/google/gemini-3.1-pro-preview" \
    material_api_key_env=GEMINI_API_KEY \
    synthesis_model="openrouter/google/gemini-3-flash-preview" \
    synthesis_api_key_env=GEMINI_API_KEY \
    linker_model="openrouter/google/gemini-3.1-pro-preview" \
    linker_api_key_env=GEMINI_API_KEY \
    plot_model="openrouter/anthropic/claude-sonnet-4.6" \
    output_dir="results/" \
    with_performance=true
```

#### Advanced: Batch processing with selective models

```bash
# Process all papers in folder with custom models and domain filtering
lemat-synth batch papers/ \
    synthesis_model=anthropic/claude-sonnet-4-6 \
    material_model=gemini/gemini-2.5-pro \
    judge_model=anthropic/claude-opus-4-7 \
    domain=catalysis \
    skip_existing=true \
    max_papers_parallel=2 \
    output_dir="results/catalysis/"
```

---

## `lemat-synth batch` — folder of papers

```
lemat-synth batch INPUT_DIR [key=value ...]
```

Processes every `.txt`, `.md`, and `.pdf` file in `INPUT_DIR` and writes
per-paper results to `output_dir/`.

### Examples

```bash
# Basic — processes all papers in folder
lemat-synth batch papers/

# Quick test run (first 5 papers only)
lemat-synth batch papers/ max_papers=5

# Custom output folder and domain filtering
lemat-synth batch papers/ \
    output_dir=results/catalysis/ \
    domain=catalysis

# Re-process everything (skip_existing=false)
lemat-synth batch papers/ skip_existing=false

# Reduce parallelism to avoid rate limits
lemat-synth batch papers/ max_papers_parallel=2

# Use Mistral OCR for all PDFs
lemat-synth batch papers/ pdf_extractor=mistral

# Powerful models, catalysis domain, resume if interrupted
lemat-synth batch papers/ \
    synthesis_model=gemini/gemini-2.5-pro \
    material_model=anthropic/claude-opus-4-7 \
    domain=catalysis \
    skip_existing=true \
    max_papers_parallel=2

# Different models through OpenRouter
lemat-synth batch papers/ \
    synthesis_model=openrouter/google/gemini-3-flash-preview \
    material_model=openrouter/google/gemini-3.1-pro-preview \
    judge_model=openrouter/anthropic/claude-sonnet-4-6 \
    api_base=https://openrouter.ai/api/v1
```

---

## Configuration Details

All arguments in the [Quick Reference table](#quick-reference-all-arguments) above can be overridden from the command line. Defaults are read from `config/cli.yaml`.

### Model strings

Model strings follow the [LiteLLM](https://docs.litellm.ai/docs/providers)
convention: `{provider}/{model-name}`.  Common providers and models:

```
gemini/gemini-2.0-flash                           # Google Gemini
gemini/gemini-2.5-pro
anthropic/claude-sonnet-4-6                      # Anthropic Claude
anthropic/claude-opus-4-7
openai/gpt-4o                                    # OpenAI
openai/gpt-4o-mini
mistral/mistral-large                            # Mistral
openrouter/google/gemini-3-flash-preview          # OpenRouter (requires api_base + key)
openrouter/google/gemini-3.1-pro-preview
openrouter/anthropic/claude-sonnet-4.6
openrouter/deepseek/deepseek-v3.2
openrouter/qwen/qwen3.5-35b-a3b
openrouter/moonshotai/kimi-k2.5
```

When using OpenRouter models, **always** set `api_base=https://openrouter.ai/api/v1`.

### API key environment variables

By default LiteLLM auto-detects API keys from standard environment variables:
- `gemini/*` → `GEMINI_API_KEY`
- `anthropic/*` → `ANTHROPIC_API_KEY`
- `openai/*` → `OPENAI_API_KEY`
- etc.

Use the `*_api_key_env` arguments to override this — useful for OpenRouter key slots or when multiple keys exist for the same provider.

**Allowed env var names:**
`ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `OPENAI_API_KEY`, `MISTRAL_API_KEY`,
`OPENROUTER_QWEN_API_KEY`, `OPENROUTER_KIMI_API_KEY`, `OPENROUTER_DEEPSEEK_API_KEY`

```bash
# Example: different OpenRouter keys for different models
lemat-synth batch papers/ \
    synthesis_model=openrouter/qwen/qwen3.5-35b-a3b \
    synthesis_api_key_env=OPENROUTER_QWEN_API_KEY \
    linker_model=openrouter/moonshotai/kimi-k2.5 \
    linker_api_key_env=OPENROUTER_KIMI_API_KEY \
    api_base=https://openrouter.ai/api/v1
```

### Synthesis extraction modes

| Mode | Description | Usage |
|------|-------------|-------|
| Default | Single LLM extraction (synthesis_model) | `lemat-synth extract paper.txt` |
| Domain-specific | Filters plots to domain-relevant figures | `domain=catalysis` or `electrochemistry` |
| With performance | Extracts and links plot data to materials | `with_performance=true` (requires Claude API) |
| Custom prompts | Override extraction instructions | `prompts.synthesis_instructions="..."` |

### PDF and figure processing

| Argument | Options | When to use |
|----------|---------|-------------|
| `pdf_extractor` | `docling` (default) | Local, no API key required |
| | `mistral` | Better for scanned/low-quality PDFs (requires MISTRAL_API_KEY) |
| `figure_segmenter` | `dino` (default) | Fast, 28-class detection |
| | `florence` | More accurate, binary quantitative/qualitative classification |
| `florence_repo_id` | HuggingFace repo ID | LoRA adapter for Florence (only when `figure_segmenter=florence`) |

### Domain filtering (when `with_performance=true`)

| Domain | Figures kept | Use case |
|--------|--------------|----------|
| `generic` | All figures (no filtering) | Default for multi-domain papers |
| `catalysis` | Conversion/selectivity vs temperature curves | Catalysis materials |
| `superconductors` | Resistivity ρ(T) and resistance R(T) plots | Superconductor data |
| `electrochemistry` | Current/capacitance vs voltage curves | Battery/electrochemistry materials |

### Batch processing options

| Argument | Default | Purpose |
|----------|---------|---------|
| `max_papers` | `null` | Stop after N papers (useful for test runs) |
| `skip_existing` | `true` | Resume from last run; set to `false` to reprocess all |
| `max_papers_parallel` | `4` | Concurrent papers; lower if hitting rate limits |

---

## Customizing Prompts

Every prompt used during extraction can be customized. The full set of
prompt keys is in `config/cli.yaml` under the `prompts:` block:

| Prompt | Purpose |
|--------|---------|
| `prompts.synthesis_system` | System message for synthesis extraction |
| `prompts.synthesis_instructions` | Task instructions for synthesis extraction |
| `prompts.paper_text_description` | Description of the input paper text |
| `prompts.material_name_description` | Description of the target material |
| `prompts.synthesis_output_description` | Description of the output structure |
| `prompts.material_instructions` | Task instructions for material extraction |
| `prompts.material_input_description` | Description of material input |
| `prompts.material_output_description` | Description of material output |

### Override from command line

Quote the value to preserve spaces and special characters:

```bash
# Focus on specific synthesis methods
lemat-synth extract paper.txt \
    "prompts.synthesis_instructions=Extract only sol-gel synthesis procedures. \
    Ignore characterization and testing sections."

# Customize material name handling
lemat-synth extract paper.txt \
    "prompts.material_name_description=The specific compound formula to extract, \
    including all dopants and promoters."
```

### Permanent changes

Edit `config/cli.yaml` directly to make changes that apply to all future runs.

---

## Environment Variables

Add these to your `.env` file (automatically loaded at runtime):

| Variable | When required | Example use |
|----------|---------------|-|
| `GEMINI_API_KEY` | Using `gemini/*` models | Default synthesis/material models |
| `ANTHROPIC_API_KEY` | Using Claude models or `with_performance=true` | `synthesis_model=anthropic/claude-sonnet-4-6` |
| `OPENAI_API_KEY` | Using `openai/gpt-*` models | `plot_model=openai/gpt-4o` |
| `MISTRAL_API_KEY` | Using Mistral models or `pdf_extractor=mistral` | `pdf_extractor=mistral` for better OCR |
| `OPENROUTER_QWEN_API_KEY` | Using Qwen via OpenRouter | `synthesis_model=openrouter/qwen/qwen3.5-35b-a3b` |
| `OPENROUTER_KIMI_API_KEY` | Using Kimi via OpenRouter | `linker_model=openrouter/moonshotai/kimi-k2.5` |
| `OPENROUTER_DEEPSEEK_API_KEY` | Using DeepSeek via OpenRouter | `synthesis_model=openrouter/deepseek/deepseek-v3.2` |

See [API key selection (per component)](#api-key-environment-variables) for examples of using multiple keys simultaneously.

---

## Managing Concurrency & Rate Limits

Two independent settings control parallel API calls:

| Setting | Default | To reduce rate limits |
|---------|---------|----------------------|
| Papers processed in parallel (batch mode) | 4 | `max_papers_parallel=2` |
| LLM calls per paper (async operations) | env-driven | `LLM_SYNTHESIS_MAX_CONCURRENT_LLM_CALLS=4` in `.env` |

If you hit rate-limit errors, reduce one or both values:

```bash
# Reduce papers processed concurrently
lemat-synth batch papers/ max_papers_parallel=2

# Reduce concurrent API calls per paper
# Add to .env: LLM_SYNTHESIS_MAX_CONCURRENT_LLM_CALLS=4
```

---

## Output structure

Results are written to `output_dir/<paper-name>/`.  Each folder contains
one JSON file per extracted material, plus optional performance files.

See the [Output Format](output-format.md) page for a full description of
the JSON schema.

---

## When to use the Hydra deployment scripts instead

The `lemat-synth` CLI covers the most common case — extracting from your
own papers.  For advanced workflows, use the scripts in
`examples/scripts/deployment/` directly:

- **Multi-LLM ensemble extraction** (`synthesis_extraction=multi_llm`)
- **Processing the full HuggingFace LeMat-Synth-Papers dataset**
- **Evaluation against human annotations** (`data_loader=annotation`)
- **Full Hydra sweep / multi-run** mode

Those scripts are configured with `examples/config/config.yaml` and
support the same Hydra override syntax.
