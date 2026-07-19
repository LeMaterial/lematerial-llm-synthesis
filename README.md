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
```
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
   - Recommended: `pip install -U "huggingface_hub"`
   - Or (macOS): `brew install hf`
3. **Login with access token**: `hf auth login`

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

End-to-end workflow: extract synthesis + performance data from NH₃ decomposition papers, evaluate VLM extraction accuracy against human-annotated ground truth.

**Files** — [`examples/scripts/case_study_thermocatalysis/`](examples/scripts/case_study_thermocatalysis/)

| File | Purpose |
|---|---|
| [`run.py`](examples/scripts/case_study_thermocatalysis/run.py) | **Single entry point** — extraction + multi-VLM eval |
| [`eval_vlm.py`](examples/scripts/case_study_thermocatalysis/eval_vlm.py) | RMSE/MAE vs. human GT (imported by `run.py`) |
| [`catalysis_map.py`](examples/scripts/case_study_thermocatalysis/catalysis_map.py) | Generates 7 publication figures from batch results |
| [`run_case_study.sh`](examples/scripts/case_study_thermocatalysis/run_case_study.sh) | **Full walkthrough script** — runs all phases end-to-end |
| [`catalysis_synthesis_with_performance.ipynb`](examples/notebooks/catalysis_synthesis_with_performance.ipynb) | Interactive single-paper pipeline |

**Ground truth** — [`data/results_catalysis_human/`](data/results_catalysis_human/) (26 papers, 170 materials — **read-only**).

#### Prerequisites

```bash
# API keys in .env at repo root
ANTHROPIC_API_KEY=...
GEMINI_API_KEY=...
MISTRAL_API_KEY=...
OPENROUTER_QWEN_API_KEY=...       # for Qwen VLMs
OPENROUTER_DEEPSEEK_API_KEY=...   # for DeepSeek VLMs
```

PDFs in `data/papers_catalysis/`. Only 2 PDFs have matching ground truth (`Teng_2024_Ru` → `Teng2024Ru`, `Zhou_2021_...` → `zhou2021`); use `--match-gt-only` to restrict to those.

#### Quickstart — full run (one script)

```bash
bash examples/scripts/case_study_thermocatalysis/run_case_study.sh
```

Edit the `VLMS=()` array at the top to select which models to benchmark. Runs all three phases automatically.

#### Two-phase workflow (recommended for multi-VLM benchmarking)

Synthesis extraction (OCR + materials + synthesis + figure detection) is slow (~30 min/paper) and VLM-independent. Cache it once, then the VLM step (~5 min/paper) is the only thing that runs per model:

```bash
cd examples/scripts/case_study_thermocatalysis

# Phase 1 (once) — saves to data/results_cache/_cache/<paper_id>/
python run.py \
    --pdf-dir   ../../../data/papers_catalysis \
    --output    ../../../data/results_cache \
    --match-gt-only \
    --phase     synthesis \
    --no-eval \
    --skip-existing

# Phase 2 (per VLM) — reads cache, runs VLM extraction + linking
python run.py \
    --output    ../../../data/results_catalysis/claude-sonnet-4.6 \
    --phase     vlm \
    --cache     ../../../data/results_cache \
    --vlms      claude-sonnet-4.6 \
    --single-dir

# Repeat for each additional VLM — no re-extraction needed
python run.py \
    --output    ../../../data/results_catalysis/gemini-3-flash \
    --phase     vlm \
    --cache     ../../../data/results_cache \
    --vlms      gemini-3-flash \
    --single-dir
```

Cache layout:
```
data/results_cache/_cache/
    Teng_2024_Ru/
        synthesis.json   ← materials + synthesis + paper text
        figures.json     ← detected figures with base64 image data
    Zhou_2021_.../
        synthesis.json
        figures.json
```

#### Eval — compare all VLMs to ground truth

```bash
python run.py \
    --output  ../../../data/results_catalysis \
    --gt      ../../../data/results_catalysis_human \
    --vlms    claude-sonnet-4.6 gemini-3-flash gpt-4o \
    --eval-only \
    --metric  rmse \
    --csv     ../../../data/results_catalysis/ranking.csv
```

Prints a ranked RMSE table, saves `vlm_ranking_rmse.json` + CSV. RMSE=0 perfect; 0.02–0.15 good; >0.3 poor.

#### Visualize results

After extraction, generate 7 publication-quality figures:

```bash
python examples/scripts/case_study_thermocatalysis/catalysis_map.py \
    data/results_catalysis/claude-sonnet-4.6 \
    --out-dir data/results_catalysis/claude-sonnet-4.6/figures
```

Outputs PNG + PDF for: conversion landscape, metal/support heatmap, synthesis network, radar charts, promoter analysis, conversion-by-synthesis, 3D waterfall + `landscape_data.csv`.

Optional flags: `--use-llm` (LLM-assisted material name parsing), `--ref-temp 500` (reference temperature), `--debug` (data inventory).

#### Output layout

```
data/results_catalysis/
    <vlm_name>/
        <paper_id>/
            <material>.json            ← synthesis procedure + plot_data coordinates
            performance_mappings.json  ← which plot series → which material
            linking_summary_llm.json   ← linking stats + quality scores
            batch_summary.json         ← run timing + material counts
        figures/                       ← catalysis_map.py output (PNG/PDF)
    manifest.json                      ← which PDFs ran + GT mapping
    vlm_ranking_rmse.json              ← VLM ranking by mean RMSE
    ranking.csv                        ← per-material scores for all VLMs
```

Each `<material>.json`:
```json
{
  "material": "Ru/MgO(110)",
  "synthesis": { "...synthesis procedure..." },
  "performance": {
    "material_name": "Ru/MgO(110)",
    "plot_data": [{
      "series_name": "Ru/MgO(110)",
      "coordinates": [[T, conversion], ...],
      "x_axis_label": "Temperature", "x_axis_unit": "°C",
      "y_axis_label": "NH3 conversion", "y_axis_unit": "%"
    }]
  }
}
```

#### All flags

| Flag | Purpose |
|---|---|
| `--phase synthesis\|vlm\|all` | Two-phase mode (default: `all`) |
| `--cache PATH` | Cache dir for phase 1 (required for `--phase vlm`) |
| `--match-gt-only` | Only process PDFs with a matching GT folder |
| `--skip-existing` | Skip papers already processed |
| `--max N` | Process only first N papers (testing) |
| `--eval-only` | Skip extraction, only run eval |
| `--single-dir` | Treat `--output` as flat results dir (no VLM subdir) |
| `--no-eval` | Skip GT comparison |
| `--metric rmse\|mae` | Error metric (default: `rmse`) |
| `--csv PATH` | Export per-material scores to CSV |

#### Available VLMs

Any key from `LLM_REGISTRY` in [`src/llm_synthesis/utils/llms.py`](src/llm_synthesis/utils/llms.py):

| Key | Model |
|---|---|
| `claude-sonnet-4.6` | Anthropic Claude Sonnet 4.6 |
| `gemini-3-flash` | Google Gemini 3 Flash |
| `gpt-4o` | OpenAI GPT-4o |
| `qwen3.5-397b-a17b` | Qwen via OpenRouter |
| `deepseek-v3.2` | DeepSeek via OpenRouter |
| `gemini-2.5-flash` | Google Gemini 2.5 Flash |
| `mistral-medium` | Mistral Medium |

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

#### Putting it together — the porous materials case study

The porosity case study is a concrete example of a custom domain built with these primitives. Here is how it is wired up in [`examples/scripts/case_study_porosity/run.py`](examples/scripts/case_study_porosity/run.py):

**Step 1 — define what plots to keep**

Adsorption isotherms have pressure on the x-axis and uptake/loading on the y-axis. The filter matches both axis labels and units, and vetoes plots about temperature, heat, or selectivity (which share some keywords but are not isotherms):

```python
from llm_synthesis.config.plot_filter_config import PlotFilterConfig

plot_filter = PlotFilterConfig(
    x_axis_labels=[
        "pressure", "p", "p/p0", "p/p₀", "relative pressure",
        "p (bar)", "p (kpa)", "p (mpa)", "p (atm)", "p (pa)",
        "p [bar]", "p [kpa]", "p [atm]",
    ],
    x_axis_units=["bar", "kpa", "mpa", "atm", "pa", "p0", "p/p0"],
    y_axis_keywords=[
        "loading", "uptake", "adsorption", "coverage", "surface area",
        "amount adsorbed", "quantity adsorbed",
        "cm³/g", "cm3/g", "mmol/g", "mol/kg", "mg/g", "wt%", "cc/g",
    ],
    y_axis_units=[
        "mmol/g", "mol/kg", "cm³/g", "cm3/g", "cc/g", "mg/g",
        "wt%", "ml/g", "l/g", "g/g", "mmol g⁻¹", "cm³ g⁻¹",
    ],
    y_axis_exclude_patterns=[
        "temperature", "time", "heat", "enthalpy",
        "selectivity", "permeability", "diffusivity",
    ],
    require_y_keyword_with_percentage=False,
)
```

**Step 2 — tell the material extractor what to look for**

Porous materials papers study MOFs, zeolites, COFs, and related frameworks. Each variant (different linker, metal node, or activation condition) should be a separate entry:

```python
material_extraction_instructions = (
    "Extract ALL distinct porous or framework material compositions "
    "that were synthesized and characterized in this paper. "
    "If the paper studies multiple variants (e.g., different linkers, "
    "metal nodes, activation conditions) list EACH variant separately. "
    "Focus on materials for which adsorption or porosity data are reported."
)

material_output_description = (
    "ALL distinct synthesized porous material compositions as a "
    "comma-separated list using chemical formulas or IUPAC names. "
    "Include variant labels where relevant (e.g., 'MOF-5-activated', "
    "'ZIF-8-NH2')."
)
```

**Step 3 — choose an output writer**

No domain-specific scalars are needed beyond what the pipeline already extracts, so both optional extractors are `None`. Results are saved as per-material JSON files with annotation templates using `AnnotatedJsonWriter`:

```python
from llm_synthesis.runners.output_writers.json_writer import AnnotatedJsonWriter

output_writer = AnnotatedJsonWriter()
```

**Step 4 — assemble `DomainConfig` and run**

```python
from llm_synthesis.config.domain_config import DomainConfig
from llm_synthesis.runners.batch_runner import BatchRunner

domain = DomainConfig(
    name="porosity",
    plot_filter_config=plot_filter,
    material_extraction_instructions=material_extraction_instructions,
    material_output_description=material_output_description,
    text_metric_extractor=None,
    vlm_metric_processor=None,
    output_writer=output_writer,
)

runner = BatchRunner(
    domain_config=domain,
    gemini_model="gemini-3.0-flash",
    claude_model="claude-sonnet-4-20250514",
    material_model="gemini-3.0-pro",
    synthesis_max_tokens=80_000,
    linker_max_tokens=32_000,
)
runner.run(pdf_dir="/path/to/pdfs", output_dir="/path/to/results")
```

This is exactly what `DomainConfig.for_porosity()` returns — the factory method is just a convenience wrapper around the same four steps above.

Run it:
```bash
python examples/scripts/case_study_porosity/run.py \
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
### Human Annotation App

<details>
<summary><b>Streamlit annotator for scoring extractor outputs</b></summary>

Run from repo root:

```bash
streamlit run examples/scripts/data_curation/annotator_app.py
```

**Workflow:**
1. Select paper ID
2. Open/read PDF in app
3. Fill or update `human_recipe`
4. Score each extractor tab
5. Save → `annotations/<paper_id>/result_human.json`

**Submit annotations:**
```bash
git add annotations/<paper_id>/result_human.json
git commit -m "annotate/<paper_id>"
git push
```

Or open a dedicated PR:
```bash
git fetch origin
git checkout -b annotate/<paper_id> origin/main
git add annotations/<paper_id>/result_human.json
git commit -m "annotate/<paper_id>"
git push -u origin annotate/<paper_id>
gh pr create --fill
```

> If `uv sync` fails on your platform: `pip install "streamlit==1.55.0"`

</details>

### Evaluate LLM-as-Judge Results

<details>
<summary><b>Agreement metrics: judge vs. human, judge ranking, extractor quality</b></summary>

Once you have `annotations/<paper_id>/{result.json,result_human.json}` pairs (extraction + judge scores vs. human ground truth), run the evaluation scripts from `examples/scripts/evaluation/`:

```bash
# Judge ranking + synth-LLM x judge-LLM heatmap
python examples/scripts/evaluation/compare_multi_llm_results_complete.py --rank-by abs_diff

# Agreement broken down by material category
python examples/scripts/evaluation/compare_multi_llm_results_by_category.py

# Judge/extractor insight tables (self-preference, LOO ranking, dimension means, ...)
python examples/scripts/evaluation/analyze_judge_extractor_insights.py
```

All outputs (CSVs, JSON, PNG heatmaps) are written to `results/agreement_analysis/`. See [examples/scripts/evaluation/README.md](examples/scripts/evaluation/README.md) for the full evaluation design, available metrics, and how to use the results to pick an extraction/judge LLM.

</details>

### Customize LeMat-Synth
*Work in Progress*
{EXAMPLES HOW TO GENERALIZE/ABSTRACT EXTRACTION PIPELINE}

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
