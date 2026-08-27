---
title: LeMat-Synth
description: >-
  An open-source multi-modal toolbox for extracting structured synthesis
  procedures and performance data from materials science literature at scale.
hide:
  - navigation
  - toc
---

<div class="ls-hero" markdown>

![LeMaterial](assets/lematerial-logo-light.png#only-light){ .ls-hero__logo }
![LeMaterial](assets/lematerial-logo.png#only-dark){ .ls-hero__logo }

# LeMat-Synth

<p class="ls-hero__tagline">
Turn materials science papers into machine-readable science — synthesis
recipes, performance curves and quality scores, extracted at scale by LLMs and
VLMs.
</p>

[Get started](getting-started/quickstart.md){ .md-button .md-button--primary }
[Tutorials](tutorials/index.md){ .md-button }
[Dataset](getting-started/dataset.md){ .md-button }

<p class="ls-hero__badges">
<a href="https://arxiv.org/abs/2510.26824"><img alt="Paper" src="https://img.shields.io/badge/arXiv-2510.26824-b31b1b.svg"></a>
<a href="https://huggingface.co/datasets/LeMaterial/LeMat-Synth"><img alt="Dataset" src="https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Dataset-yellow"></a>
<a href="https://github.com/LeMaterial/lematerial-llm-synthesis"><img alt="GitHub" src="https://img.shields.io/badge/GitHub-source-181717?logo=github"></a>
<a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/python-3.11%2B-blue"></a>
<a href="https://github.com/LeMaterial/lematerial-llm-synthesis/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/badge/license-Apache%202.0-green"></a>
</p>

</div>

---

## What you get

<div class="grid cards ls-features" markdown>

-   :material-flask-outline:{ .lg .middle } __Synthesized materials__

    ---

    Every material a paper actually makes, identified by chemical formula and
    separated from the ones it merely cites.

    [:octicons-arrow-right-24: How extraction works](developer-guide/architecture.md)

-   :material-format-list-numbered:{ .lg .middle } __Structured procedures__

    ---

    Step-by-step recipes — reagents, amounts, temperatures, durations,
    atmospheres — normalised into a controlled Pydantic ontology.

    [:octicons-arrow-right-24: Output format](user-guide/output-format.md)

-   :material-chart-line:{ .lg .middle } __Performance data__

    ---

    Quantitative values read straight off the paper's plots by a VLM, then
    linked back to the material each curve belongs to.

    [:octicons-arrow-right-24: Case studies](case-studies/index.md)

-   :material-check-decagram-outline:{ .lg .middle } __A quality score__

    ---

    An LLM judge rates every extraction on five dimensions, benchmarked against
    human annotators so you know what to trust.

    [:octicons-arrow-right-24: Annotations & evaluation](developer-guide/annotations.md)

</div>

This site documents the reference implementation of
[LeMat-Synth v1.0](https://arxiv.org/abs/2510.26824) (NeurIPS AI4Mat 2025) — and
the extensible codebase underneath it, for building extraction studies in your
own domain.

![The LeMat-Synth pipeline: papers in, structured synthesis and performance data out](assets/overview.png){ .ls-figure }

---

## Get started in a minute

=== "Install"

    Requires **Python 3.11+** and
    [uv](https://github.com/astral-sh/uv?tab=readme-ov-file#installation).

    ```bash
    git clone https://github.com/LeMaterial/lematerial-llm-synthesis.git
    cd lematerial-llm-synthesis

    uv venv -p 3.11 --seed
    uv sync && uv pip install -e .

    cp .env.example .env
    echo "GEMINI_API_KEY=your_key_here" >> .env
    ```

    A free [Gemini API key](https://aistudio.google.com/app/apikey) is enough for
    the default pipeline. Add `ANTHROPIC_API_KEY` to read data off plots, and
    `MISTRAL_API_KEY` for OCR on scanned PDFs.

    [:octicons-arrow-right-24: Full installation guide](getting-started/installation.md)

=== "Command line"

    ```bash
    # One paper
    lemat-synth extract paper.pdf

    # A whole folder, with performance curves read from the figures
    lemat-synth batch papers/ output_dir=results/ \
        domain=catalysis with_performance=true
    ```

    Results land in `results/<paper_id>/<material>.json`.

    [:octicons-arrow-right-24: CLI reference](user-guide/cli.md)

=== "Python"

    ```python
    from llm_synthesis.transformers.synthesis_extraction.dspy_synthesis_extraction import (
        DspySynthesisExtractor,
        make_dspy_synthesis_extractor_signature,
    )
    from llm_synthesis.utils import clean_text
    from llm_synthesis.utils.dspy_utils import get_llm_from_name

    extractor = DspySynthesisExtractor(
        signature=make_dspy_synthesis_extractor_signature(
            instructions="Extract the complete synthesis procedure for this material."
        ),
        lm=get_llm_from_name("gemini-2.0-flash"),
    )

    synthesis = extractor.forward(input=(clean_text(paper_text), "Fe2O3"))
    print(synthesis.synthesis_method, synthesis.steps)
    ```

    [:octicons-arrow-right-24: Python API guide](user-guide/python-api.md)

=== "Just the data"

    The extracted corpus is already published — no API key, no cost:

    ```python
    from datasets import load_dataset

    synth = load_dataset("LeMaterial/LeMat-Synth", split="train")
    ```

    The dataset is gated on HuggingFace: request access once, then `hf auth login`.

    [:octicons-arrow-right-24: Dataset access](getting-started/dataset.md)

!!! tip "Do a small run first"

    Cost scales with the number of **materials**, not papers. Start with
    `max_papers=5` before pointing the pipeline at a whole corpus.

??? example "What a result file looks like"

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

    Every field is explained in [Output Format](user-guide/output-format.md).

---

## Learn by running

Seven notebooks take you from *reading* LeMat-Synth data to *producing* it, and
finally to *changing what gets produced*. Each states its API keys and cost up
front, and runs unchanged locally or on Google Colab.

<div class="grid cards" markdown>

-   :material-database-search-outline:{ .lg .middle } __Use the data__

    ---

    Explore the published dataset and assemble a paper corpus of your own —
    HuggingFace access only, no cost.

    *Tutorials 1–2*

-   :material-cog-play-outline:{ .lg .middle } __Extract__

    ---

    Run the CLI over a folder, pull synthesis *and* performance out of a single
    paper, and score the result with the judge.

    *Tutorials 3–5*

-   :material-puzzle-outline:{ .lg .middle } __Extend__

    ---

    Change the ontology, then point the whole pipeline at a scientific domain it
    has never seen.

    *Tutorials 6–7*

</div>

[Browse the tutorials](tutorials/index.md){ .md-button .md-button--primary }

---

## Case studies

Three domain studies ship with the repository, each a thin script over the
shared `DomainConfig` + `BatchRunner` pair.

<div class="ls-wide-table" markdown>

| Domain | Extracts | |
|---|---|---|
| **Thermocatalysis** | Synthesis + NH₃-conversion curves, with a multi-VLM benchmark against human ground truth | [Read](case-studies/thermocatalysis.md) |
| **Superconductors** | Synthesis + *T*<sub>c</sub>, read from text *and* geometrically from ρ(T) plots | [Read](case-studies/superconductors.md) |
| **Porous materials** | Synthesis + adsorption isotherms for MOFs, zeolites and COFs | [Read](case-studies/porosity.md) |

</div>

Building a fourth — electrochemistry, battery cycling, thermoelectrics — means
assembling four pieces and handing them to `BatchRunner`. You never edit the
pipeline.

[Build your own case study](case-studies/custom-domain.md){ .md-button }

---

## Find your way around

<div class="ls-wide-table" markdown>

| I want to… | Go to |
|---|---|
| Install and run my first extraction | [Quickstart](getting-started/quickstart.md) |
| Use the published data instead of extracting | [Dataset Access](getting-started/dataset.md) |
| See every CLI setting | [CLI Reference](user-guide/cli.md) |
| Understand a result file | [Output Format](user-guide/output-format.md) |
| Build a pipeline in Python | [Python API](user-guide/python-api.md) · [API Reference](api/pipeline.md) |
| Switch LLMs, or run dataset-scale jobs | [Configuration & Models](developer-guide/configuration.md) |
| Change what gets extracted | [Architecture](developer-guide/architecture.md) |
| Contribute or use ground-truth annotations | [Annotations](developer-guide/annotations.md) |
| Fix something that broke | [Troubleshooting](user-guide/troubleshooting.md) |

</div>

---

## Contributing

Contributions are welcome — new extractors, new domains, and especially **new
human annotations**, which are what make quality measurable. Read
[CONTRIBUTING.md](https://github.com/LeMaterial/lematerial-llm-synthesis/blob/main/CONTRIBUTING.md)
for the workflow, or start the annotator app:

```bash
streamlit run examples/scripts/data_curation/annotator_app.py
```

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

Released under the [Apache License 2.0](https://github.com/LeMaterial/lematerial-llm-synthesis/blob/main/LICENSE).
