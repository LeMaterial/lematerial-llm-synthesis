# Superconductors

Extracts synthesis procedures and critical temperatures (*T*<sub>c</sub>) from
superconductor papers. *T*<sub>c</sub> is read twice — once from the text, and
once geometrically from ρ(T) or R(T) plots — which gives a built-in
cross-check on both readings.

**Source** — [`examples/scripts/case_study_superconductors/`](https://github.com/LeMaterial/lematerial-llm-synthesis/tree/main/examples/scripts/case_study_superconductors)

| Script | What it does |
|---|---|
| `keyword_search.py` | Filters `LeMat-Synth-Papers` by the *Superconductor* category and the keyword `resistivity` → `results/db_superconductors.pkl` |
| `downsample_with_llm.py` | Gemini pass that keeps only papers with a genuine ρ(T)/R(T) plot |
| `run.py` | Standard batch run — synthesis + text *T*<sub>c</sub> + VLM *T*<sub>c</sub> → `tc_master.csv` |
| `batch_run_tc.py` | Earlier standalone *T*<sub>c</sub> runner, kept for reproducing published numbers |
| `batch_run_tc_new_snippet.py` | Adds a bottom-left crop ("snippet") VLM pass for hard-to-read plots → `tc_master_snippet.csv` |

Matching exploratory notebooks live in
[`examples/notebooks/dev/`](https://github.com/LeMaterial/lematerial-llm-synthesis/tree/main/examples/notebooks/dev):
`superconductivity_tc_extraction.ipynb` (single-paper walkthrough),
`superconductivity_tc_extraction_plus_snippet.ipynb` (same, with the snippet
pass), `visualisation_tc.ipynb` (*T*<sub>c</sub>-vs-year, text/VLM agreement,
synthesis-method breakdowns) and
`visualisation_tc_with_human_annotation.ipynb` (the same plots against human
annotations).

> [!NOTE]
> Notebooks under `examples/notebooks/dev/` are working copies: they are kept
> runnable but are not maintained to the same standard as the
> [tutorials](../tutorials/index.md).

---

## Step 1 — Build a corpus

Screen the published paper dataset down to plausible candidates:

```bash
uv run examples/scripts/case_study_superconductors/keyword_search.py
```

Filters on the *Superconductor* category field and the keyword `resistivity` in
abstracts, writes `results/db_superconductors.pkl`, and opens a pull request on
HuggingFace with the filtered subset.

## Step 2 — Downsample with an LLM

Requires `GEMINI_API_KEY`. Verifies each paper actually contains a ρ(T) or R(T)
plot rather than a pure field-sweep study:

```bash
# Concise prompt
uv run examples/scripts/case_study_superconductors/downsample_with_llm.py --prompt default

# Detailed prompt with explicit magnetic-field exclusion rules (recommended)
uv run examples/scripts/case_study_superconductors/downsample_with_llm.py --prompt long
```

Pushes the filtered list to HuggingFace and downloads up to 100 sample PDFs.

## Step 3 — Extract

```bash
uv run examples/scripts/case_study_superconductors/run.py <pdf_dir> <output_dir> \
    --skip-existing
```

Writes one JSON per paper plus a growing `tc_master.csv` — one row per
(paper, material) with both the text-derived and plot-derived *T*<sub>c</sub>.

Flags: `--max N` (first *N* papers only), `--skip-existing` (resume an
interrupted run), `--skip-figures` (text-only, no VLM — much faster and cheaper).

Defaults: `gemini-3.0-flash` for synthesis and linking, `gemini-3.0-pro` for
material extraction, `claude-sonnet-4-20250514` for plot reading. Edit the
constants at the top of `run.py` to change them.

<details>
<summary><b>Snippet-based extraction for hard plots</b></summary>

Some ρ(T) curves drop to zero in a small corner of a busy multi-panel figure.
`batch_run_tc_new_snippet.py` adds a second VLM pass over a bottom-left crop of
each plot, which recovers the transition when the full-figure read misses it:

```bash
uv run examples/scripts/case_study_superconductors/batch_run_tc_new_snippet.py \
    /path/to/superconductor_pdfs --skip-existing
```

Outputs `<pdf_folder>/results_snippet/tc_master_snippet.csv`.

</details>

---

## What makes this domain different

The domain config is
[`DomainConfig.for_superconductivity()`](https://github.com/LeMaterial/lematerial-llm-synthesis/blob/main/src/llm_synthesis/config/domain_config.py),
which differs from the generic pipeline in three ways:

1. **Plot filter** — `PlotFilterConfig.for_superconductivity()` keeps resistance
   and resistivity against temperature, and **vetoes** anything mentioning
   *field*, so magnetoresistance panels never reach the VLM.
2. **Material prompt** — asks for each doping level and stoichiometry as a
   separate entry, because *T*<sub>c</sub> is what varies between them.
3. **A domain metric processor** — an extra VLM pass that locates the
   superconducting transition on the curve, rather than just digitising it.

That third piece is a `BaseVLMMetricProcessor`; see
[Building your own case study](custom-domain.md#3-domain-metric-extractors-optional)
for how to write one.
