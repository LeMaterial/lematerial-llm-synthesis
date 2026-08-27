<div align="center">

<img src="assets/lematerial-logo.png" alt="LeMaterial" width="420">

# LeMat-Synth

**An open-source multi-modal toolbox for extracting structured synthesis procedures
and performance data from materials science literature — at scale.**

[![Paper](https://img.shields.io/badge/arXiv-2510.26824-b31b1b.svg)](https://arxiv.org/abs/2510.26824)
[![Dataset](https://img.shields.io/badge/🤗%20HuggingFace-Dataset-yellow)](https://huggingface.co/datasets/LeMaterial/LeMat-Synth)
[![Docs](https://img.shields.io/badge/docs-lematerial.github.io-informational)](https://lematerial.github.io/lematerial-llm-synthesis/)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)

[Documentation](https://lematerial.github.io/lematerial-llm-synthesis/) ·
[Quickstart](https://lematerial.github.io/lematerial-llm-synthesis/getting-started/quickstart/) ·
[Tutorials](https://lematerial.github.io/lematerial-llm-synthesis/tutorials/) ·
[Dataset](https://huggingface.co/datasets/LeMaterial/LeMat-Synth) ·
[Paper](https://arxiv.org/abs/2510.26824)

</div>

<br>

Give LeMat-Synth a paper — PDF or text — and it returns machine-readable science:

- **Synthesized materials**, identified by chemical formula
- **Step-by-step synthesis procedures**, structured into a controlled Pydantic ontology
- **Quantitative performance data**, read off the paper's plots and linked back to the material it belongs to
- **A quality score per extraction**, from an LLM judge benchmarked against human annotators

This repository is the reference implementation of
[LeMat-Synth v1.0](https://arxiv.org/abs/2510.26824) (NeurIPS AI4Mat 2025), plus
an extensible codebase for building your own domain-specific extraction studies.

![Pipeline overview](assets/overview.png)

---

## Install

Requires **Python 3.11+** and [uv](https://github.com/astral-sh/uv?tab=readme-ov-file#installation).

```bash
git clone https://github.com/LeMaterial/lematerial-llm-synthesis.git
cd lematerial-llm-synthesis

uv venv -p 3.11 --seed
uv sync && uv pip install -e .

uv run python -c "import llm_synthesis"   # prints nothing if it worked
```

Then add your API keys to `.env`:

```bash
cp .env.example .env
echo "GEMINI_API_KEY=your_key_here" >> .env
```

A free [Gemini API key](https://aistudio.google.com/app/apikey) is enough for the
default pipeline. Add `ANTHROPIC_API_KEY` for reading data off plots, and
`MISTRAL_API_KEY` for OCR on scanned PDFs.

> [!TIP]
> The `lemat-synth` CLI and the notebooks load `.env` themselves — there is
> nothing to `source`. Full options, including Windows and OpenRouter setups, are in
> the [Installation guide](https://lematerial.github.io/lematerial-llm-synthesis/getting-started/installation/).

---

## Sixty-second start

```bash
# One paper
lemat-synth extract paper.pdf

# A whole folder, with performance curves read from the figures
lemat-synth batch papers/ output_dir=results/ domain=catalysis with_performance=true
```

Results land in `results/<paper_id>/<material>.json`:

<details>
<summary><b>What a result file looks like</b></summary>

```json
{
  "material": "Ru/MgO(110)",
  "synthesis": {
    "target_compound": "Ru/MgO(110)",
    "target_compound_type": "functional materials & catalysts",
    "synthesis_method": "wet impregnation",
    "starting_materials": [
      {"name": "RuCl3·xH2O", "amount": 0.21, "unit": "g", "purity": "99.9%"}
    ],
    "steps": [
      {"step_number": 1, "action": "dissolve",
       "conditions": {"temperature": 25, "temp_unit": "C"}},
      {"step_number": 2, "action": "calcine",
       "conditions": {"temperature": 500, "temp_unit": "C",
                      "duration": 4, "time_unit": "h", "atmosphere": "Ar"}}
    ]
  },
  "performance": {
    "plot_data": [{
      "series_name": "Ru/MgO(110)",
      "coordinates": [[350, 12.4], [400, 41.9], [450, 78.2]],
      "x_axis_label": "Temperature", "x_axis_unit": "°C",
      "y_axis_label": "NH3 conversion", "y_axis_unit": "%"
    }]
  },
  "evaluation": {
    "scores": {"overall_score": 4.2, "structural_completeness_score": 4.5},
    "confidence_level": "high"
  }
}
```

Every field is explained in
[Output Format](https://lematerial.github.io/lematerial-llm-synthesis/user-guide/output-format/).

</details>

> [!IMPORTANT]
> Always do a `max_papers=5` run first. Cost scales with the number of
> **materials**, not papers — a folder of catalysis papers can be several times
> more expensive than the same number of single-material papers.

---

## Or skip extraction entirely

The extracted data is already published. If someone has already run the pipeline
over the papers you care about, reading the dataset costs nothing and needs no
API key:

```python
from datasets import load_dataset

synth = load_dataset("LeMaterial/LeMat-Synth", split="train")
```

Both datasets are gated on HuggingFace — request access once, then
`hf auth login`. See [Dataset Access](https://lematerial.github.io/lematerial-llm-synthesis/getting-started/dataset/).

---

## Tutorials

Seven runnable notebooks in
[`examples/notebooks/tutorials/`](examples/notebooks/tutorials/), each stating
its API keys and cost up front. Every one runs **locally or on Google Colab** —
same file, no edited cells.

| # | Tutorial | Track | API keys | Cost |
|---|----------|-------|----------|------|
| 1 | [Explore the LeMat-Synth dataset](examples/notebooks/tutorials/01_explore_the_lemat_synth_dataset.ipynb) | Use the data | HuggingFace only | Free |
| 2 | [Finding papers](examples/notebooks/tutorials/02_finding_papers.ipynb) | Use the data | HuggingFace only | Free |
| 3 | [Batch extraction with the CLI](examples/notebooks/tutorials/03_batch_extraction_with_the_cli.ipynb) | Extract | Gemini *or* OpenRouter | Fractions of a cent |
| 4 | [Synthesis + performance from a paper](examples/notebooks/tutorials/04_extracting_synthesis_and_performance.ipynb) | Extract | Gemini + Anthropic, *or* OpenRouter | $0.10–0.40 |
| 5 | [Evaluating extraction quality](examples/notebooks/tutorials/05_evaluating_extraction_quality.ipynb) | Extract | Gemini *or* OpenRouter | ~Free |
| 6 | [Customizing the ontology](examples/notebooks/tutorials/06_customizing_the_ontology.ipynb) | Extend | None | Free |
| 7 | [Building a custom case study](examples/notebooks/tutorials/07_building_a_custom_case_study.ipynb) | Extend | Gemini *or* OpenRouter | Under $0.01 |

```bash
uv run jupyter lab examples/notebooks/tutorials/
```

Every tutorial that calls a model has a `USE_OPENROUTER` flag — set it to `True`
to route all its calls through a single `OPENROUTER_API_KEY` instead of
per-provider keys. Prerequisites and the full `.env` reference are in the
[Tutorials documentation](https://lematerial.github.io/lematerial-llm-synthesis/tutorials/).

---

## Case studies

Three domain-specific studies ship with the repository, each a thin script over
the shared `DomainConfig` + `BatchRunner` pair:

| Domain | Extracts | Guide |
|---|---|---|
| **Thermocatalysis** | Synthesis + NH₃-conversion curves, with a multi-VLM benchmark against human ground truth | [Docs](https://lematerial.github.io/lematerial-llm-synthesis/case-studies/thermocatalysis/) |
| **Superconductors** | Synthesis + *T*<sub>c</sub>, read from text *and* geometrically from ρ(T) plots | [Docs](https://lematerial.github.io/lematerial-llm-synthesis/case-studies/superconductors/) |
| **Porous materials** | Synthesis + adsorption isotherms for MOFs, zeolites and COFs | [Docs](https://lematerial.github.io/lematerial-llm-synthesis/case-studies/porosity/) |

Building a fourth — electrochemistry, battery cycling, thermoelectrics — means
assembling four pieces and handing them to `BatchRunner`; you never edit the
pipeline. [Tutorial 7](examples/notebooks/tutorials/07_building_a_custom_case_study.ipynb)
builds one from scratch in a notebook, and the
[Building your own case study](https://lematerial.github.io/lematerial-llm-synthesis/case-studies/custom-domain/)
guide is the matching API reference.

> [!NOTE]
> `data/` is git-ignored, so no PDFs or ground-truth files ship with the
> repository. Tutorial 2 shows how to assemble a corpus of your own.

---

## Documentation

| I want to… | Go to |
|---|---|
| Install and run my first extraction | [Quickstart](https://lematerial.github.io/lematerial-llm-synthesis/getting-started/quickstart/) |
| Use the published data instead of extracting | [Dataset Access](https://lematerial.github.io/lematerial-llm-synthesis/getting-started/dataset/) |
| See every CLI setting | [CLI Reference](https://lematerial.github.io/lematerial-llm-synthesis/user-guide/cli/) |
| Understand a result file | [Output Format](https://lematerial.github.io/lematerial-llm-synthesis/user-guide/output-format/) |
| Build a pipeline in Python | [Python API](https://lematerial.github.io/lematerial-llm-synthesis/user-guide/python-api/) |
| Switch LLMs, or run dataset-scale jobs | [Configuration & Models](https://lematerial.github.io/lematerial-llm-synthesis/developer-guide/configuration/) |
| Change what gets extracted | [Architecture](https://lematerial.github.io/lematerial-llm-synthesis/developer-guide/architecture/) · [Tutorial 6](examples/notebooks/tutorials/06_customizing_the_ontology.ipynb) |
| Contribute or use ground-truth annotations | [Annotations](https://lematerial.github.io/lematerial-llm-synthesis/developer-guide/annotations/) |
| Fix something that broke | [Troubleshooting](https://lematerial.github.io/lematerial-llm-synthesis/user-guide/troubleshooting/) |

Build the docs locally with `uv run mkdocs serve`.

---

## Contributing

Contributions are welcome — new extractors, new domains, and especially **new
human annotations**, which are what make quality measurable.

Read [CONTRIBUTING.md](CONTRIBUTING.md) for the workflow. In short: branch off
`main`, use [Conventional Commits](https://www.conventionalcommits.org/)
(`feat:`, `fix:`, `docs:`), run `uvx pre-commit install` once so `ruff` and
`nbstripout` run on every commit, and open a PR.

To contribute an annotation, run the annotator app and submit the result:

```bash
streamlit run examples/scripts/data_curation/annotator_app.py
```

See the [Annotations guide](https://lematerial.github.io/lematerial-llm-synthesis/developer-guide/annotations/)
for the full workflow.

---

## Citation

If you use LeMat-Synth in your research, please cite:

```bibtex
@article{lederbauer2025lemat,
  title={LeMat-Synth: a multi-modal toolbox to curate broad synthesis procedure
         databases from scientific literature},
  author={Lederbauer, Magdalena and Betala, Siddharth and Li, Xiyao and
          Jain, Ayush and Sehaba, Amine and Channing, Georgia and
          Germain, Gr{\'e}goire and Leonescu, Anamaria and Flaifil, Faris and
          Amayuelas, Alfonso and Nozadze, Alexandre and Schmid, Stefan P. and
          Zaki, Mohd and Ethirajan, Sudheesh Kumar and Pan, Elton and
          Franckel, Mathilde and Duval, Alexandre and Krishnan, N. M. Anoop and
          Gleason, Samuel P.},
  journal={arXiv preprint arXiv:2510.26824},
  year={2025}
}
```

---

## License

Apache License 2.0 — see [LICENSE](LICENSE).
