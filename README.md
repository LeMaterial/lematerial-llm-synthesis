![](assets/lematerial-logo.png)

# LeMaterial-Synthesis

An open-source multi-modal toolbox for extracting structured synthesis procedures and performance data from materials science literature at scale. This repository contains the implementations of [LeMat-Synth v1.0](https://arxiv.org/abs/2510.26824) (published on the arXiv and presented at NeurIPS AI4Mat 2025) plus the extendable codebase for usecases in materials science.

![](assets/overview.png)

[![Paper](https://img.shields.io/badge/arXiv-2512.04562-b31b1b.svg)](https://arxiv.org/abs/2510.26824)
[![Dataset](https://img.shields.io/badge/🤗%20HuggingFace-Dataset-yellow)](https://huggingface.co/datasets/LeMaterial/LeMat-Synth)

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




### Thermocatalysis Case Study

Extracts synthesis procedures and catalytic performance data (conversion/selectivity vs temperature curves) from a local corpus of heterogeneous catalysis papers (PDFs not part of the open-source LeMat-Synth-Papers corpus).

**Scripts** — [`examples/scripts/case_study_thermocatalysis/`](examples/scripts/case_study_thermocatalysis/)

| Script / Notebook | What it does |
|---|---|
| `run_all_papers.py` | Full synthesis + performance extraction on a local folder of PDFs → per-paper JSON results |
| `catalysis_synthesis_with_performance.ipynb` | Step-by-step interactive extraction for a single paper |
| `catalysis_map_notebook.ipynb` | Visualizations: conversion landscape, per-metal subplots |
| `keyword_search.py` | *(Experimental)* Keyword filtering of LeMat-Synth-Papers — not used in the main pipeline |
| `downsample_with_llm.py` | *(Experimental)* LLM screening for performance-vs-temperature plots — not used in the main pipeline |

**Run extraction** on your local PDF corpus:
```bash
uv run examples/scripts/case_study_thermocatalysis/run_all_papers.py \
  /path/to/catalysis_corpus \
  /path/to/results_catalysis/ \
  --skip-existing
```
For each paper the script saves:
- `<output_dir>/<paper_id>/<material>.json` — synthesis procedure + evaluation score per material
- `<output_dir>/<paper_id>/performance_mappings.json` — plot series linked to materials
- `<output_dir>/<paper_id>/linking_summary_llm.json` — LLM quality evaluation
- `<output_dir>/<paper_id>/linking_summary_human.json` — blank template for human annotation
- `<output_dir>/batch_summary.json` — overall batch statistics

Additional flags: `--max N` to limit to the first N papers, `--skip-existing` to resume an interrupted run.

**Explore results interactively:**
- Open `catalysis_synthesis_with_performance.ipynb` to walk through every extraction step on a single paper (PDF → materials → synthesis → figures → plot data → linking).
- Open `catalysis_map_notebook.ipynb` to produce publication-quality conversion landscape and per-metal subplot figures from the batch results.

---

### Superconductor Case Study

Extracts synthesis procedures and critical temperatures (Tc) from superconductor papers using both text extraction and vision-language model (VLM) reading of ρ(T)/R(T) plots.

**Scripts** — [`examples/scripts/case_study_superconductors/`](examples/scripts/case_study_superconductors/)

| Script / Notebook | What it does |
|---|---|
| `keyword_search.py` | Filters LeMat-Synth-Papers by "Superconductor" category + "resistivity" keyword → `results/db_superconductors.pkl` |
| `downsample_with_llm.py` | Gemini LLM check for ρ(T)/R(T) plots → filtered dataset on HuggingFace + sample PDFs |
| `batch_run_tc.py` | Full Tc extraction (text + VLM) on PDFs → `tc_master.csv` + per-paper JSONs |
| `batch_run_tc_new_snippet.py` | Enhanced extraction: adds bottom-left crop (snippet) VLM pass + synthesis extraction → `tc_master_snippet.csv` |
| `superconductivity_tc_extraction.ipynb` | Step-by-step interactive extraction for a single paper |
| `superconductivity_tc_extraction_plus_snippet.ipynb` | Same as above with additional snippet-based VLM extraction |
| `visualisation_tc.ipynb` | Visualizations: Tc vs year, text/VLM agreement, synthesis methods |
| `visualisation_tc_with_human_annotation.ipynb` | Same + comparison against human-annotated ground truth |

**Step 1 — Keyword + category filtering** (screens LeMat-Synth-Papers on HuggingFace):
```bash
uv run examples/scripts/case_study_superconductors/keyword_search.py
```
Filters by the "Superconductor" category field and the keyword `"resistivity"` in abstracts. Outputs `results/db_superconductors.pkl` and creates a PR on HuggingFace with the filtered subset.

**Step 2 — LLM downsampling** (requires `GEMINI_API_KEY`):
```bash
# Concise prompt
uv run examples/scripts/case_study_superconductors/downsample_with_llm.py --prompt default

# Detailed prompt with explicit magnetic-field exclusion rules (recommended)
uv run examples/scripts/case_study_superconductors/downsample_with_llm.py --prompt long
```
Uses Gemini to verify each paper contains a ρ(T) or R(T) plot that is not purely a field-sweep study. Pushes the filtered list to HuggingFace and downloads up to 100 sample PDFs.

**Step 3 — Extract Tc from PDFs:**

Standard extraction (text extraction + VLM figure reading):
```bash
uv run examples/scripts/case_study_superconductors/batch_run_tc.py /path/to/superconductor_pdfs
```
Outputs `<pdf_folder>/results/tc_master.csv` with one row per (paper, material).

Enhanced extraction (adds snippet-based VLM crop + synthesis extraction):
```bash
uv run examples/scripts/case_study_superconductors/batch_run_tc_new_snippet.py \
  /path/to/superconductor_pdfs \
  --skip-existing
```
Outputs `<pdf_folder>/results_snippet/tc_master_snippet.csv`.

Additional flags for both batch scripts: `--max N` to limit to the first N papers, `--skip-existing` to resume an interrupted run, `--skip-figures` for text-only mode (no VLM, faster).

**Explore results interactively:**
- Open `superconductivity_tc_extraction.ipynb` for a guided single-paper walkthrough.
- Open `superconductivity_tc_extraction_plus_snippet.ipynb` for the same with the snippet VLM pass.
- Open `visualisation_tc.ipynb` to produce Tc-vs-year scatter plots, text/VLM agreement plots, and synthesis method breakdowns.
- Open `visualisation_tc_with_human_annotation.ipynb` to compare pipeline output against human-annotated ground truth.

### Adding Your Own Case Study

The pipeline is built around four composable pieces. Once you configure them, `BatchRunner` handles everything else — PDF loading, SI detection, rate-limit retries, progress reporting, and output.

```
┌────────────────────────────────────────────────────────────────┐
│                         BatchRunner                            │
│  ┌──────────────┐   ┌───────────────────────────────────────┐  │
│  │ DomainConfig │──▶│    SynthesisPerformancePipeline       │  │
│  │              │   │  (PDF → materials → synthesis →       │  │
│  │ PlotFilter   │   │   figures → plot data → linking)      │  │
│  │ MatPrompt    │   └───────────────────────────────────────┘  │
│  │ TextExtract? │──▶  Optional: BaseTextMetricExtractor        │
│  │ VLMProcess?  │──▶  Optional: BaseVLMMetricProcessor         │
│  │ OutputWriter │──▶  AnnotatedJsonWriter / CsvMasterWriter    │
│  └──────────────┘                                              │
└────────────────────────────────────────────────────────────────┘
```

#### The four pieces

**1. `PlotFilterConfig` — which plots are relevant?**

Every paper produces many figures. `PlotFilterConfig` filters them by axis labels and units so only domain-relevant plots reach the downstream LLM. For thermocatalysis you want temperature on the x-axis and conversion/yield on the y-axis; for superconductors you want resistance vs. temperature.

```python
from llm_synthesis.config.plot_filter_config import PlotFilterConfig

# Use a built-in preset:
PlotFilterConfig.for_catalysis()        # T vs. conversion/yield
PlotFilterConfig.for_superconductivity() # R(T) / ρ(T) plots
PlotFilterConfig.for_coverage()          # P vs. adsorption isotherms

# Or build a custom one:
filter_cfg = PlotFilterConfig(
    x_axis_labels=["current density", "j"],
    x_axis_units=["ma/cm2", "a/cm2"],
    y_axis_keywords=["overpotential", "faradaic efficiency"],
    y_axis_units=["%", "mv"],
    filter_x_axis=True,
    filter_y_axis=True,
)
```

Key fields:
| Field | Purpose |
|---|---|
| `x_axis_labels` | Substrings matched (case-insensitive) against the x-axis label |
| `x_axis_units` | Exact unit strings that also signal a relevant x-axis |
| `y_axis_keywords` | Substrings matched against the y-axis label |
| `y_axis_units` | Unit strings that signal a relevant y-axis |
| `y_axis_exclude_patterns` | Veto list — overrides keyword matches (e.g. exclude "field" for superconductors) |
| `filter_x_axis` / `filter_y_axis` | Set either to `False` to skip that axis entirely |

**2. Material extraction prompt — what to look for in the text?**

Two strings tell the material extractor what to find:

- `material_extraction_instructions` — free-text instructions to the LLM (be specific about variants, dopings, loadings).
- `material_output_description` — describes the expected output format (e.g. "comma-separated chemical formulas including loading percentages").

The quality of downstream synthesis and performance linking depends heavily on these strings. Be explicit about whether you want each doping level listed separately, whether precursors should be included, etc.

**3. Domain metric extractors (optional) — domain-specific scalars**

After the standard pipeline runs, you can attach two optional extraction passes:

`BaseTextMetricExtractor` — one extra LLM call on the paper text:
```python
from llm_synthesis.domain_metrics.base import BaseTextMetricExtractor

class MyTextExtractor(BaseTextMetricExtractor):
    def extract(self, paper_text: str, materials: list[str]) -> dict:
        # Call your LLM, parse the output.
        # Return: {material_name: {metric_key: value, ...}, ...}
        return {
            "Cu0.5Ba0.5": {"bandgap_eV": 1.8, "measured_at_K": 300},
        }
```

`BaseVLMMetricProcessor` — one extra VLM pass on the already-filtered relevant plots:
```python
from llm_synthesis.domain_metrics.base import BaseVLMMetricProcessor

class MyVLMProcessor(BaseVLMMetricProcessor):
    def process(
        self,
        relevant_plots,   # list[(index, ExtractedLinePlotData)]
        plot_figures,     # list[FigureInfo] — contains the image bytes
        plot_mappings,    # series-to-material links from the pipeline
        materials,        # list[str]
        paper_text,       # full paper text for context
    ) -> dict:
        # Run your VLM calls, return per-material metrics.
        return {
            "Cu0.5Ba0.5": {"onset_current_mA": 12.3},
        }
```

Both return the same shape: `{material_name: {key: value}}`. Set either to `None` in `DomainConfig` to skip it.

**4. `BaseOutputWriter` — how to save results**

Two built-in writers cover most needs:

| Writer | Best for | Output |
|---|---|---|
| `AnnotatedJsonWriter` | Qualitative / rich data | `<output_dir>/<paper_id>/<material>.json` + linking summaries |
| `CsvMasterWriter` | Tabular data you want to aggregate across papers | Same JSON + a growing `master.csv` with one row per (paper, material) |

For a fully custom CSV schema, subclass `CsvMasterWriter` and override `_build_flat_records`:
```python
from llm_synthesis.runners.output_writers.csv_writer import CsvMasterWriter

MY_COLUMNS = ["paper_id", "material", "bandgap_eV", "synthesis_method"]

class BandgapWriter(CsvMasterWriter):
    def __init__(self):
        super().__init__(csv_columns=MY_COLUMNS, master_csv_name="bandgaps.csv")

    def _build_flat_records(self, paper_id, result, text_metrics, vlm_metrics):
        records = []
        for entry in result.results:
            m = entry.material
            records.append({
                "paper_id": paper_id,
                "material": m,
                "bandgap_eV": text_metrics.get(m, {}).get("bandgap_eV"),
                "synthesis_method": (
                    entry.synthesis.synthesis_method if entry.synthesis else None
                ),
            })
        return records
```

#### Putting it together — minimal new case study

```python
# examples/scripts/case_study_electrochemistry/run.py
from llm_synthesis.config.domain_config import DomainConfig
from llm_synthesis.config.plot_filter_config import PlotFilterConfig
from llm_synthesis.runners.batch_runner import BatchRunner
from llm_synthesis.runners.output_writers.json_writer import AnnotatedJsonWriter

domain = DomainConfig(
    name="electrochemistry",
    plot_filter_config=PlotFilterConfig(
        x_axis_labels=["potential", "voltage", "v vs"],
        x_axis_units=["v", "mv"],
        y_axis_keywords=["current", "faradaic efficiency", "overpotential"],
        y_axis_units=["%", "ma", "mv"],
        filter_x_axis=True,
        filter_y_axis=True,
    ),
    material_extraction_instructions=(
        "Extract ALL distinct electrocatalyst compositions synthesized and "
        "tested. List each loading, dopant level, and substrate variant "
        "separately (e.g. '1%Pt/C', '3%Pt/C', 'IrO2/Ti')."
    ),
    material_output_description=(
        "Comma-separated chemical formulas including loading percentages "
        "and substrate (e.g. '1%Pt/C, 3%Pt/C, IrO2/Ti')."
    ),
    text_metric_extractor=None,   # add a BaseTextMetricExtractor subclass here
    vlm_metric_processor=None,    # add a BaseVLMMetricProcessor subclass here
    output_writer=AnnotatedJsonWriter(),
)

runner = BatchRunner(
    domain_config=domain,
    gemini_model="gemini-3.0-flash",
    claude_model="claude-sonnet-4-20250514",
)
runner.run(pdf_dir="/path/to/pdfs", output_dir="/path/to/results")
```

Run it:
```bash
python examples/scripts/case_study_electrochemistry/run.py \
  /path/to/pdfs /path/to/results --skip-existing
```

#### Using a built-in domain config

For the three built-in domains you don't need to construct `DomainConfig` manually:

```python
DomainConfig.for_catalysis()
DomainConfig.for_porosity()
DomainConfig.for_superconductivity(claude_model="claude-sonnet-4-20250514")
```

#### Decision guide

| I want to... | What to change |
|---|---|
| Filter different plot types | Customize `PlotFilterConfig` |
| Extract different materials | Edit `material_extraction_instructions` / `material_output_description` |
| Pull a scalar from the text (e.g. bandgap, yield) | Implement `BaseTextMetricExtractor` |
| Read a value geometrically from a figure | Implement `BaseVLMMetricProcessor` |
| Save results as a growing CSV | Use `CsvMasterWriter` (or subclass for custom columns) |
| Save rich per-material JSON with annotation templates | Use `AnnotatedJsonWriter` |
| Skip figure extraction entirely (faster, text only) | Pass `--skip-figures` or `skip_figures=True` to `runner.run()` |

---

## 📝 Citation

Cite us:

```bibtex
@article{lederbauer2026mapping,
  title={Mapping Materials Science: a multi-modal toolbox to curate broad synthesis procedure databases from scientific literature},
  author={WIP},
  journal={WIP},
  year={2026}
}
```

```bibtex
@article{lederbauer2025lemat,
  title={LeMat-Synth: a multi-modal toolbox to curate broad synthesis procedure databases from scientific literature},
  author={Lederbauer, Magdalena and Betala, Siddharth and Li, Xiyao and Jain, Ayush and Sehaba, Amine and
          Channing, Georgia and Germain, Gr{\'e}goire and Leonescu, Anamaria and Flaifil, Faris and
          Amayuelas, Alfonso and Nozadze, Alexandre and Schmid, Stefan P. and Zaki, Mohd
          and Ethirajan, Sudheesh Kumar and Pan, Elton and Franckel, Mathilde
          and Duval, Alexandre and Krishnan, N. M. Anoop and Gleason, Samuel P.},
  journal={arXiv preprint arXiv:2510.26824},
  year={2025}
}
```
