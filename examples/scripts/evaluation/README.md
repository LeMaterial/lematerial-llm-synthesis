# Annotation Evaluation & Agreement Metrics

This guide explains how the `result.json` / `result_human.json` annotations are evaluated, which metrics are available, how to run the analyses, and how to use the outputs to choose an **extraction** LLM and a **judge** LLM.

## The evaluation design (what the data represents)

Each paper folder under `annotations/<id>/` holds two files:

- `result.json` — the multi-LLM extraction + judge matrix. It is a list of entries, one per **extraction LLM** (`synth_llm`). Each entry has `materials[]`; each material has the extracted `synthesis` recipe plus an `evaluations[]` list with one entry per **judge LLM** (`judge_llm`). So the file encodes an **N extractors × M judges** grid (currently 4 × 4: `claude-sonnet-4.6`, `gemini-3-flash`, `qwen3.5-397b-a17b`, `deepseek-v3.2`).
- `result_human.json` — the human ground truth. It holds (a) the corrected `human_recipe` per material, and (b) the **human judge's** `evaluations[]`: one score entry **per extraction LLM**, index-matched to the top-level `extractor_order` list. The mapping is purely positional and matches `result.json`'s `synth_llm` order — `evaluations[0]` is the human's score of `extractor_order[0]`'s extraction, `[1]` of `extractor_order[1]`, and so on. There is **one human judge** rating each of the N LLM extractions (not N separate human evaluators).

So the human is effectively an **(N+1)th judge** whose row is the reference the LLM judges are measured against. Every evaluation (human or LLM) produces the same 8 scores, each on a **1.0–5.0** scale.

## Metrics available

### A. Judge score dimensions (the raw scores each evaluation produces)

Defined in `src/llm_synthesis/metrics/judge/general_synthesis_judge.py` (`GeneralSynthesisEvaluationScore`) and mirrored by `SCORE_COLUMNS` in `eval_utils.py`. Each is 1.0–5.0 (0.5 increments) with a paired `_reasoning` string:

- `structural_completeness_score`
- `material_extraction_score`
- `process_steps_score`
- `equipment_extraction_score`
- `conditions_extraction_score`
- `semantic_accuracy_score`
- `format_compliance_score`
- `overall_score` (recomputed by the judge as the mean of the 7 criteria)

The judge also emits `confidence_level` (low/medium/high), `missing_information`, `extraction_errors`, and `improvement_suggestions` (not scored, but useful qualitative signal).

### B. Agreement metrics (how well two evaluators agree)

Computed by `compute_agreement_metrics()` in `eval_utils.py`. Given the human judge's score vector and an LLM judge's score vector **over the same (extractor, material) cells**, it returns:

- **Spearman rho** + **p-value** — rank correlation (permutation-based p, `n_resamples=10_000`, for small samples; asymptotic otherwise). Measures whether the judge *ranks* extractions the same way humans do.
- **Quadratic-weighted Cohen's kappa** — agreement on binned ordinal categories (scores are bucketed 0–4 via `categorize_score`). Penalizes larger disagreements more.
- **ICC(2,1)** (`calculate_icc_absolute_agreement`) — two-way random-effects, **absolute agreement**: do the judge's scores match the human's *values*, not just their ranking.
- **ICC(3,1)** (`calculate_icc_consistency`) — two-way mixed-effects, **consistency**: do they agree up to a constant offset.
- **mean_diff** and **abs_diff** — signed and absolute mean score gap (bias and magnitude of disagreement). `abs_diff` is the default ranking metric.
- Descriptive stats: human/LLM **mean, median, std**, and sample size **n**.

Rule of thumb: `abs_diff` (lower = closer to human), `rho`/`kappa`/ICC (higher = better agreement).

### C. Non-LLM structural metrics (exact-match checks)

In `src/llm_synthesis/metrics/text_extraction/structured_synthesis.py` — cheap binary {0,1} checks comparing an extracted ontology to a reference:

- `NumberCheckerMetric` — same number of steps.
- `MaterialsCheckerMetric` — same set of starting materials.
- `TargetCheckerMetric` — same target compound.

These are used inside the extraction pipeline / DSPy metrics rather than the agreement scripts, but they are the objective (non-judge) signal available.

### D. Related judges (separate subsystems, same metric style)

- **Linking judge** (`linking_judge.py`, `linking_evaluation_ontology.py`): scores `material_identity_score`, `performance_data_correctness_score`, `completeness_score`, `format_structure_score`, `overall_score` (1–5) plus 9 boolean failure flags (F1–F9). Used for the synthesis↔performance linking step, not recipe extraction.
- **Plot/figure extraction** (`eval_utils.series_coord_metrics`): `RMSE_norm`, `MAE_norm`, Pearson r, Spearman rho, ICC(2,1) against a ground-truth CSV. Only relevant to figure/plot extraction.

## The evaluation scripts

All live in `examples/scripts/evaluation/`. Both rank the **multi-LLM** judges against the human judge for the current `result.json` / `result_human.json` layout. Outputs go under `results/agreement_analysis/`.

| Script | Question it answers | Reads | Key outputs |
|---|---|---|---|
| `compare_multi_llm_results_complete.py` | Which judge LLM agrees best with the human overall? How does each (extractor × judge) pair compare? | `annotations/<id>/result.json` + `result_human.json` | `results/agreement_analysis/multi_llm_complete.log`, `multi_llm_judge_ranking.json`, `multi_llm_judge_ranking.png`, `multi_llm_heatmap_synth_judge.png` |
| `compare_multi_llm_results_by_category.py` | How does judge agreement vary by `target_compound_type` / `synthesis_method`, and per extractor/judge? | same | `multi_llm_by_category.log`, `multi_llm_agreement_by_material_category.csv`, `multi_llm_heatmap_*.png` |

`compare_multi_llm_results_by_category.py` reuses `load_annotations()` from `compare_multi_llm_results_complete.py`; both share `eval_utils.py`. They align human and LLM materials by fuzzy name match (`similarity_threshold=0.7`).

> Two older scripts (`compare_human_judge_scores_complete.py` / `_by_category.py`) were removed: they only read the now-deleted `annotations/<id>/old/` layout and handled a single judge. The multi-LLM scripts supersede them — the human is included as one judge row, and all LLM judges are compared to it at once.

### Commands

```bash
# Multi-LLM judge ranking (the main analysis). Rank metric: abs_diff | rho | kappa | icc2 | icc3
uv run python examples/scripts/evaluation/compare_multi_llm_results_complete.py --rank-by abs_diff

# Per-category multi-LLM breakdown
uv run python examples/scripts/evaluation/compare_multi_llm_results_by_category.py
```

Both scripts accept `--annotations-dir` (default `annotations/`).

## How to decide: extraction LLM vs judge LLM

These are two different questions answered from the same `result.json` grid.

**Choosing the extraction LLM (`synth_llm`)** — you want the extractor whose recipes score best:
- **Most direct signal:** the human judge already scored every extractor. Read the human `overall_score` (and per-criterion scores) from `result_human.json`, mapped to each extractor via `extractor_order`, averaged across materials/papers. Higher = the human rated that extractor's recipes better.
- **Corroborating signal:** in `result.json`, aggregate the LLM-judge `overall_score` **grouped by `synth_llm`**, averaged across judges and papers. The `synth_judge_heatmap` (rows = synth LLM) in `compare_multi_llm_results_complete.py` visualizes this. Trust it more when the judges themselves agree with the human (see below).
- Also weigh cost (`cost_report.json` from the extraction run) and schema-validity of that model's extractions.

**Choosing the judge LLM (`judge_llm`)** — you want the judge whose scores best track human judgement:
- Run `compare_multi_llm_results_complete.py`. It ranks each judge by agreement with the human judge's scores (over the same extractor/material cells). Pick the judge with the **lowest `abs_diff`** and **highest `rho` / `kappa` / ICC**. `multi_llm_judge_ranking.json` is the machine-readable ranking.
- Use `compare_multi_llm_results_by_category.py` to confirm the choice is stable across material types / synthesis methods (a judge can be great on ceramics but poor on polymers).
- A good judge is one you can then trust to score *new* extractions where no human label exists.

In short: **judge quality is measured against humans; extraction quality is measured by (good) judges.** Establish the best judge first, then use it to compare extractors.

## Generating `result.json` (the multi-LLM run)

Configure the model lists in `examples/config/synthesis_extraction/multi_llm.yaml` and `examples/config/judge/multi_llm.yaml`, then:

```bash
uv run examples/scripts/deployment/extract_synthesis_multi_llm_judge.py \
  data_loader=local \
  data_loader.architecture.data_dir="/path/to/markdown" \
  synthesis_extraction=multi_llm \
  material_extraction=multi_llm \
  judge=multi_llm \
  result_save=multi_llm
```

Outputs land in `results/single_run/<timestamp>/`: `result.json`, `evaluation_matrix.png`, `global_avg_evaluation_matrix.png`, `cost_report.json`. Available models are registered in `LLM_REGISTRY` (`src/llm_synthesis/utils/llms.py`). See `examples/scripts/deployment/README_multi_llm.md`.

## Caveats & gotchas

- **Hardcoded skip-lists may be stale.** Both scripts hardcode skips for `annotation_guide_catalysis`, `f2f0828a…`, `2883daff…`, `90233593…`. Those papers have since been fixed/re-annotated, so they are currently excluded unnecessarily — trim the `skip_folders` lists if you want them included.
- **The human evaluations are positional** — matched to extractors by `extractor_order` index, with no per-eval identifier. If a `result_human.json` ever has its `evaluations[]` in a different order than `extractor_order` (or a different count than the extractors), the human scores silently misalign to the wrong extractor. The maintained multi-LLM scripts assume `evaluations[idx]` ↔ `extractor_order[idx]`. (The `aggregate_human_scores_df` "average multiple human evaluators" path applies only to the old-format `compare_human_judge_scores_*` scripts, not this layout.)
- **One human judge, small samples**: there is a single human rating per (extractor, material), so per-category metrics can be noisy — check the `n` column before trusting a category result.
- **Fuzzy material matching (threshold 0.7)** can mis-align or drop materials whose human vs LLM names differ a lot; unmatched materials are excluded from metrics.
- **`overall_score` is derived** (mean of the 7 criteria, clamped to 1–5), so it is not independent of the other scores.
