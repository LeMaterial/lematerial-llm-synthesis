# Configuration Guide

This page explains how to customise LeMat-Synth without writing Python code.
You can swap models, change data sources, and adjust extraction settings purely
through command-line flags or by editing YAML files.

> **You do not need to read this page** if you are using the notebooks or the
> `lemat-synth` CLI — those have sensible defaults built in. Come back here when
> you want to run large-scale batch jobs or try different LLMs.

---

## What are the YAML files?

The deployment scripts use a configuration system called
[Hydra](https://hydra.cc). Think of it as a recipe book: each YAML file in
`examples/config/` describes one ingredient (which LLM to call, where to load papers
from, where to save results). The master file `examples/config/config.yaml` lists
which ingredient to use by default for each slot.

You never have to edit these files — you can override any value directly from the
command line when you run a script.

### What happens when you run a script?

Here is the sequence for a typical run:

```
You run: uv run python examples/scripts/deployment/extract_synthesis_procedure_from_text.py
          │
          ▼
  Hydra reads examples/config/config.yaml
  and loads the default sub-config for each slot:
    data_loader     → data_loader/default.yaml    (HuggingFace dataset)
    synthesis_extraction → synthesis_extraction/default.yaml (Gemini 2.0 Flash)
    judge           → judge/default.yaml          (Gemini 2.0 Flash judge)
    …
          │
          ▼
  Hydra merges all files into one big config object
  and passes it to the script.
          │
          ▼
  The script instantiates Python objects from the config:
    HFLoader(dataset_uri="LeMaterial/LeMat-Synth-Papers", …)
    DspySynthesisExtractor(lm=get_llm_from_name("gemini-2.0-flash"), …)
    DspyGeneralSynthesisJudge(…)
          │
          ▼
  The pipeline runs on each paper and writes results to
    results/single_run/<date>/<time>/
```

When you add `synthesis_extraction.architecture.lm.llm_name=claude-sonnet-4.6`
on the command line, Hydra intercepts it and replaces that one value *before*
the script starts — the script itself never sees the override, it just gets
the merged config.

### The config as a wiring diagram

Each YAML file wires up one component of the pipeline. You can think of it
like a plug-board: each slot (data loader, extractor, judge, …) has a socket,
and each YAML file is a plug you can swap out. The `_target_` key in each
file names the Python class that goes into that socket.

```
data_loader socket  ←→  data_loader/default.yaml  ←→  HFLoader class
synthesis socket    ←→  synthesis_extraction/default.yaml  ←→  DspySynthesisExtractor class
judge socket        ←→  judge/default.yaml         ←→  DspyGeneralSynthesisJudge class
```

Changing `data_loader=annotation` at the command line swaps the whole plug —
now `AnnotationHFLoader` goes into the data loader socket instead of `HFLoader`.

---

## The config slots

```
examples/config/
├── config.yaml               ← master file: lists the default for each slot
├── data_loader/              ← where to load papers from
│   ├── default.yaml          ← HuggingFace (LeMat-Synth-Papers dataset)
│   ├── local.yaml            ← local folder of text files
│   └── annotation.yaml       ← annotated papers (for evaluation)
├── material_extraction/      ← which LLM identifies material names
│   ├── default.yaml          ← Gemini 2.5 Flash Lite
│   └── multi_llm.yaml        ← run multiple LLMs in parallel
├── synthesis_extraction/     ← which LLM extracts synthesis procedures
│   ├── default.yaml          ← Gemini 2.0 Flash
│   └── multi_llm.yaml
├── judge/                    ← which LLM evaluates extraction quality
│   ├── default.yaml          ← Gemini 2.0 Flash
│   ├── multi_llm.yaml
│   └── linking.yaml          ← judge for plot-material linking
├── result_save/              ← where and how to save results
│   ├── default.yaml
│   └── multi_llm.yaml
└── plot_extraction/          ← settings for plot data extraction
    └── default.yaml          ← multiple VLMs, retry config, ranking metric
```

---

## Overriding settings from the command line

You never need to edit a YAML file just to change one value. Append the override
at the end of any command in `key=value` format:

```bash
# Use Claude instead of Gemini for synthesis extraction
uv run examples/scripts/deployment/extract_synthesis_procedure_from_text.py \
  synthesis_extraction.architecture.lm.llm_name=claude-sonnet-4.6

# Use local text files instead of HuggingFace
uv run examples/scripts/deployment/extract_synthesis_procedure_from_text.py \
  data_loader=local \
  data_loader.architecture.data_dir="/path/to/my/text_files"

# Change the output directory (Hydra default is a timestamped folder)
uv run examples/scripts/deployment/extract_synthesis_procedure_from_text.py \
  hydra.run.dir=my_results/run1

# Limit to 10 papers for a quick test
uv run examples/scripts/deployment/extract_synthesis_procedure_from_text.py \
  number_of_samples=10
```

To switch an entire slot to a different preset, name the preset without the `.yaml`
extension:

```bash
# Run with all default settings
uv run ... judge=default

# Run multi-LLM judge comparison
uv run ... judge=multi_llm
```

---

## Available LLM models

All models below can be used for synthesis extraction and judging.
Set the model name with
`synthesis_extraction.architecture.lm.llm_name=<name>` or
`judge.architecture.lm.llm_name=<name>`.

> **Source of truth.** The authoritative list lives in
> [`src/llm_synthesis/utils/llms.py`](../../src/llm_synthesis/utils/llms.py)
> in the `LLM_REGISTRY`. If you add a new model there, also add a row here.

| Name (use in config) | Provider | API key needed | Notes |
|---|---|---|---|
| `gemini-2.5-flash-lite` | Google | `GEMINI_API_KEY` | Fastest, cheapest; good for material extraction |
| `gemini-2.0-flash` | Google | `GEMINI_API_KEY` | **Default**; good balance of speed and quality |
| `gemini-2.5-flash` | Google | `GEMINI_API_KEY` | Better quality, slightly slower |
| `gemini-2.5-pro` | Google | `GEMINI_API_KEY` | Highest quality Gemini 2.5 model |
| `gemini-3.0-pro` | Google | `GEMINI_API_KEY` | Gemini 3 preview, used as default linker |
| `gemini-3.0-flash` | Google | `GEMINI_API_KEY` | Latest Gemini flash |
| `gemini-3.0-flash-lite` | Google | `GEMINI_API_KEY` | Latest ultra-fast Gemini model |
| `gemini-3-flash` | Google | `GEMINI_API_KEY` | Gemini 3 flash (reasoning disabled) |
| `claude-sonnet-4.6` | Anthropic | `ANTHROPIC_API_KEY` | Excellent for synthesis + plot extraction |
| `gpt-4o` | OpenAI | `OPENAI_API_KEY` | Strong general-purpose model |
| `gpt-4o-mini` | OpenAI | `OPENAI_API_KEY` | Cheaper OpenAI option |
| `gpt-4.1` | OpenAI | `OPENAI_API_KEY` | Latest OpenAI flagship |
| `gpt-o4-mini` | OpenAI | `OPENAI_API_KEY` | OpenAI o4-mini reasoning model |
| `gpt-o3-mini` | OpenAI | `OPENAI_API_KEY` | OpenAI o3-mini reasoning model |
| `mistral-small` | Mistral | `MISTRAL_API_KEY` | Mistral Small (latest) |
| `mistral-medium` | Mistral | `MISTRAL_API_KEY` | Mistral Medium (latest) |
| `mistral-large` | Mistral | `MISTRAL_API_KEY` | Good European-hosted option |
| `qwen3.5-35b-a3b` | Alibaba via OpenRouter | `OPENROUTER_QWEN_API_KEY` | Smaller Qwen open-weight model |
| `qwen3.5-397b-a17b` | Alibaba via OpenRouter | `OPENROUTER_QWEN_API_KEY` | Large open-weight model |
| `kimi-k2.5` | Moonshot via OpenRouter | `OPENROUTER_KIMI_API_KEY` | Moonshot Kimi K2.5 |
| `deepseek-v3.2` | DeepSeek via OpenRouter | `OPENROUTER_DEEPSEEK_API_KEY` | Strong reasoning model |

> **Rough cost guide** (order of magnitude, subject to change):
> - `gemini-2.5-flash-lite` / `gemini-2.0-flash`: ~$0.01–0.05 per paper
> - `gemini-2.5-flash` / `claude-sonnet-4.6` / `gpt-4o`: ~$0.05–0.20 per paper
> - `gemini-2.5-pro` / `gpt-4.1`: ~$0.20–0.50 per paper
>
> Costs depend heavily on paper length and the number of materials.
> Always test on a small batch first (`--max 5` or `number_of_samples=5`).

---

## Changing the data source

### Load from HuggingFace (default)

No changes needed. The default config loads from `LeMaterial/LeMat-Synth-Papers`.
Requires a HuggingFace account with access granted (request at the dataset page).

```bash
uv run examples/scripts/deployment/extract_synthesis_procedure_from_text.py
```

Limit the number of papers with:
```bash
uv run ... number_of_samples=50
```

### Load from local text files

Use the `local` data loader and point it to a folder containing `.txt` files.
Each file should contain the full text of one paper (one text file per paper).
A supplementary information file can be added as `<paper_name>_SI.txt`.

```bash
uv run examples/scripts/deployment/extract_synthesis_procedure_from_text.py \
  data_loader=local \
  data_loader.architecture.data_dir="/absolute/path/to/my/text_folder"
```

To convert PDFs to text first, run:
```bash
uv run examples/scripts/deployment/extract_text_from_pdfs.py
# (edit the script to set your PDF folder and output folder)
```

---

## Domain-specific plot filtering

When running the full pipeline with figure extraction, the tool filters plots to keep
only those relevant to your domain (e.g. conversion vs. temperature for catalysis).
Use the `--domain` flag:

```bash
lemat-synth batch /my/papers results/ --with-performance --domain catalysis
lemat-synth batch /my/papers results/ --with-performance --domain superconductors
lemat-synth batch /my/papers results/ --with-performance --domain electrochemistry
lemat-synth batch /my/papers results/ --with-performance --domain generic  # keep all plots
```

For the deployment scripts, the `PlotFilterConfig` object is set in the script
directly (see `extract_synthesis_with_performance.py`).

---

## I don't want to use Hydra or the CLI

If you want full programmatic control over every component, use the Python API
directly. The [quickstart notebook](../../examples/notebooks/00_quickstart.ipynb)
shows how to do this step by step — no YAML or configuration knowledge needed.

For the pipeline class API, see [docs/api/pipeline.md](../api/pipeline.md).

---

## Going deeper

If you want to understand how `_target_` works, how to add your own config
group, or how to debug the merged config before a run, see the
[Configuration Internals](../developer-guide/configuration-internals.md) page
in the Developer Guide.
