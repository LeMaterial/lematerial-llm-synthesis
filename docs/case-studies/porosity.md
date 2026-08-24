# Porous Materials

Extracts synthesis procedures and adsorption isotherms from papers on MOFs,
zeolites, COFs and related frameworks.

This is the simplest of the three case studies — no domain metric extractors, no
custom writer — which makes it the best template to copy when starting a new
domain. [Building your own case study](custom-domain.md) walks through exactly
how it is assembled.

**Source** — [`examples/scripts/case_study_porosity/run.py`](https://github.com/LeMaterial/lematerial-llm-synthesis/blob/main/examples/scripts/case_study_porosity/run.py)

---

## Run it

```bash
uv run examples/scripts/case_study_porosity/run.py <pdf_dir> <output_dir> \
    --skip-existing
```

| Flag | Effect |
|---|---|
| `--max N` | Process only the first *N* papers |
| `--skip-existing` | Resume an interrupted run |
| `--skip-figures` | Synthesis only — no isotherm extraction |

Requires `GEMINI_API_KEY`, `ANTHROPIC_API_KEY` (plot reading) and
`MISTRAL_API_KEY` (OCR). Defaults are `gemini-3.0-flash` for synthesis,
`gemini-3.0-pro` for material extraction and `claude-sonnet-4-20250514` for
plots; change the constants at the top of `run.py`.

Results land as one JSON per material under `<output_dir>/<paper_id>/`, written
by `AnnotatedJsonWriter` — see [Output Format](../user-guide/output-format.md).

An interactive version is in
[`examples/scripts/case_study_porosity/porosity_extraction_gemini.ipynb`](https://github.com/LeMaterial/lematerial-llm-synthesis/blob/main/examples/scripts/case_study_porosity/porosity_extraction_gemini.ipynb).

---

## What makes this domain different

Only two things, both of them strings and lists:

- **`PlotFilterConfig`** keeps pressure on the x-axis (`bar`, `kPa`, `p/p₀`, …)
  and uptake on the y-axis (`mmol/g`, `cm³/g`, `wt%`, …), while vetoing
  temperature, enthalpy, selectivity and permeability plots — which share
  vocabulary with isotherms but are not isotherms.
- **The material prompt** asks for each framework variant separately (different
  linker, metal node, or activation condition), because porosity is exactly what
  varies between them.

Both are reproduced in full in
[Building your own case study](custom-domain.md#putting-it-together-the-porous-materials-case-study).
