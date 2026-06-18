# The `annotations/` Directory

`annotations/` is the hand-verified ground-truth evaluation dataset that ships
with the repository. It is used to benchmark LLM extraction quality and to
drive the evaluation scripts.

---

## Why it exists

Automated extraction is only useful if you can measure how good it is.
`annotations/` provides a small, carefully curated set of papers where a human
expert has produced the correct structured output. By re-running the pipeline
on these papers and comparing the LLM output to the human reference, you get
quantitative scores for each extraction dimension (step accuracy, condition
extraction, etc.).

---

## Directory layout

```
annotations/
├── 2502.03121/          ← arXiv ID
│   ├── result.json      ← LLM-generated extraction (baseline / for comparison)
│   └── result_human.json ← human-verified ground truth
│
├── cond-mat.0603598/    ← legacy arXiv cond-mat format
│   ├── result.json
│   └── result_human.json
│
└── f2f0828a5de4a…/      ← HuggingFace hash (papers without a public arXiv ID)
    ├── result.json
    └── result_human.json
```

### Folder naming conventions

Each folder is named after the paper's identifier in one of three formats:

| Format | Example | When used |
|--------|---------|-----------|
| Modern arXiv ID | `2502.03121` | Papers from 2007 onwards |
| Legacy cond-mat arXiv | `cond-mat.0603598` | Old condensed-matter papers |
| HuggingFace hash | `f2f0828a5de4a3262edc7387…` | Papers without a public arXiv ID |

The translation between folder names and the IDs used in the HuggingFace
dataset is handled by
`src/llm_synthesis/utils/paper_id_utils.py` (`folder_id_to_hf_id` and
`hf_id_to_folder_id`).

---

## File format

### `result_human.json` — the ground truth

This is the authoritative reference file. A human expert read the paper and
filled in every field of `GeneralSynthesisOntology` by hand.

```json
{
  "schema_version": "multi_llm_v1",
  "paper_id": "cond-mat.0603598",
  "paper_url": "https://arxiv.org/pdf/cond-mat/0603598",
  "extractor_order": ["claude-sonnet-4.6", "gemini-3-flash", "qwen3.5-397b-a17b", "deepseek-v3.2"],
  "materials": [
    {
      "material_name": "LaAlO3/SrTiO3",
      "human_recipe": {
        "target_compound": "LaAlO3/SrTiO3 heterointerface",
        "target_compound_type": "two-dimensional materials",
        "synthesis_method": "pulsed laser deposition",
        "starting_materials": [ ... ],
        "steps": [ ... ],
        "equipment": [ ... ],
        "notes": null
      }
    }
  ]
}
```

| Field | Meaning |
|-------|---------|
| `schema_version` | Always `"multi_llm_v1"` for current annotations |
| `paper_id` | The folder name (matches the directory) |
| `paper_url` | Direct link to the source paper |
| `extractor_order` | Which LLMs were used to produce `result.json` |
| `materials[].material_name` | Material identifier as it appears in the paper |
| `materials[].human_recipe` | Filled `GeneralSynthesisOntology` — the reference |

`human_recipe` uses exactly the same field structure as the extraction output.
See [Output Format](../user-guide/output-format.md) for a full field-by-field
breakdown.

### `result.json` — the LLM baseline

This mirrors the pipeline's normal output format for the same paper. It is
stored alongside the human annotation so evaluation scripts can load both
files from the same directory without any path juggling.

!!! note
    `result.json` is optional. Evaluation scripts will skip a folder if
    `result_human.json` is present but `result.json` is absent, or produce
    only human-vs-human scores.

---

## How annotations are consumed

### 1. Evaluation scripts

The scripts under `examples/scripts/evaluation/` load both files for each
annotated paper, compare the LLM extraction in `result.json` against the
`human_recipe` in `result_human.json`, and compute per-dimension scores:

```bash
# Compare LLM outputs against human annotations, grouped by score category
uv run python examples/scripts/evaluation/compare_human_judge_scores_by_category.py

# Full comparison with per-paper breakdown
uv run python examples/scripts/evaluation/compare_human_judge_scores_complete.py
```

### 2. `AnnotationHFLoader` — running the pipeline on annotated papers

When you run the deployment scripts with `data_loader=annotation`, the loader
[`AnnotationHFLoader`](../api/pipeline.md) scans `annotations/` for folder
names, then fetches only those papers from the HuggingFace dataset
(`LeMaterial/LeMat-Synth-Papers`, split `sample_for_evaluation`). This lets
you benchmark any new extractor on exactly the annotated subset:

```bash
uv run python examples/scripts/deployment/extract_synthesis_procedure_from_text.py \
    data_loader=annotation \
    synthesis_extraction.architecture.lm.llm_name="claude-opus-4-7"
```

The results land in `results/` as usual and can then be compared against the
human annotations with the evaluation scripts.

---

## Adding a new annotation

Follow these steps to add a paper to the benchmark set:

**Step 1 — Choose the folder name.**
Use the paper's arXiv ID if it has one (`2502.03121`). For legacy cond-mat
papers use the dot format (`cond-mat.0603598`). For papers without an arXiv
ID, use the HuggingFace document hash from the dataset.

**Step 2 — Create the directory.**
```bash
mkdir annotations/<paper-id>
```

**Step 3 — Write `result_human.json`.**
Start from the template below, read the paper, and fill every field that the
paper actually mentions. Leave unreported fields as `null` — do **not**
invent values.

```json
{
  "schema_version": "multi_llm_v1",
  "paper_id": "<paper-id>",
  "paper_url": "<direct PDF URL>",
  "extractor_order": [],
  "materials": [
    {
      "material_name": "<formula or name>",
      "human_recipe": {
        "target_compound": "<name>",
        "target_compound_type": "<one of the 16 allowed types>",
        "synthesis_method": "<method>",
        "starting_materials": [],
        "steps": [],
        "equipment": [],
        "notes": null
      }
    }
  ]
}
```

**Step 4 — Optionally add `result.json`.**
Run the pipeline on this paper with `data_loader=annotation` (after creating
the folder) and copy the output file to `annotations/<paper-id>/result.json`.

**Step 5 — Verify.**
Run an evaluation script and confirm the new paper appears in the output:
```bash
uv run python examples/scripts/evaluation/compare_human_judge_scores_complete.py
```

---

## The `old/` sub-directory

Some annotation folders contain an `old/` sub-directory. This holds
superseded versions of `result.json` that were generated by earlier pipeline
runs or model versions. They are kept for traceability but are **not** read
by any current script.
