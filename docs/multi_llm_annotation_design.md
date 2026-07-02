# Multi-LLM Annotation Design

This document describes how to extend the annotation process from a single LLM (one extractor + one judge) to **3–4 LLMs** at extraction and at judging, so you can:

1. **Pick the best judge**: the LLM whose scores agree most with human scores.
2. **Pick the best extractor**: the LLM whose extractions score highest (by humans or by the chosen judge).
3. **Analyse bias**: whether LLMs are more lenient on their own extractions vs others’.

---

## 1. Data to store per (paper, material)

| Data | Count | Purpose |
|------|--------|---------|
| **Extractions** | 4 (one per extractor LLM) | Each LLM extracts one recipe for this material. |
| **Judge evaluations** | 4×4 = 16 | Each of 4 judges scores each of 4 extractions (including self). |
| **Human recipe** | 1 | The **one** recipe for this material, written by the annotator (same structure as the `synthesis` block in current `result_human.json`). No requirement to mirror the paper verbatim — it is the human’s reference recipe for this material. |
| **Human evaluations** | 4 | Human scores for each of the 4 LLM extractions (same schema as judge), in a **fixed order** (see below). |

Humans fill **one human recipe** per material and **4 score blocks** (one per extraction, in order). The 4 extractions and 16 LLM judge evaluations are machine-generated.

---

## 2. Proposed file layout (per paper)

Keep one folder per paper: `annotations/<paper_id>/`.

| File | Producer | Purpose |
|------|-----------|---------|
| **`result_multi.json`** | Pipeline | All 4 extractions + full 4×4 judge matrix. Single source of truth for machine data. |
| **`result_human.json`** | Annotators | **One human recipe** per material + **ordered list of 4 evaluations** (see schema below). Designed so annotators only need “my recipe, then Score 1, Score 2, Score 3, Score 4.” |

For **backward compatibility**: keep supporting `result.json` + `result_human.json` for single-LLM papers. Eval scripts can support both (single-LLM = one extractor, one judge; multi-LLM = multiple extractors, multiple judges).

---

## 3. Schema: `result_multi.json` (machine-generated)

```json
{
  "schema_version": "multi_llm_v1",
  "paper_id": "0d5ffdaf23a655e1eff80bc8b6b4978067de4d5b",
  "extractor_ids": ["claude-sonnet", "gpt-4o", "gemini-pro", "llama-3"],
  "judge_ids": ["claude-sonnet", "gpt-4o", "gemini-pro", "llama-3"],
  "materials": [
    {
      "material_name": "Fe phase nanosphere/conical CNT bundle nanostructures",
      "extractions": {
        "claude-sonnet": {
          "synthesis": { "...": "GeneralSynthesisOntology" }
        },
        "gpt-4o": {
          "synthesis": { "...": "GeneralSynthesisOntology" }
        },
        "gemini-pro": { "synthesis": { "..." } },
        "llama-3": { "synthesis": { "..." } }
      },
      "judge_evaluations": {
        "claude-sonnet": {
          "claude-sonnet": { "evaluation": { "..." } },
          "gpt-4o":      { "evaluation": { "..." } },
          "gemini-pro":  { "evaluation": { "..." } },
          "llama-3":     { "evaluation": { "..." } }
        },
        "gpt-4o": { "claude-sonnet": { "..." }, "gpt-4o": { "..." }, ... },
        "gemini-pro": { ... },
        "llama-3": { ... }
      }
    }
  ]
}
```

- **`extractor_ids` / `judge_ids`**: Fixed order for all materials (annotators and code use the same order).
- **`extractions[extractor_id]`**: Only the extracted `synthesis` (no evaluation inside).
- **`judge_evaluations[judge_id][extractor_id]`**: One `GeneralSynthesisEvaluation` (scores, reasoning, etc.) for “judge J scoring extractor E”.

So for each material the annotator sees **4 recipes** (from `result_multi.json`) and fills **4 human score blocks** in `result_human.json`.

---

## 4. Schema: `result_human.json` (annotator-filled)

Designed for **minimal cognitive load**: one human recipe per material, then **four score slots in a fixed order**. Annotators do not need to match model names to keys — “first slot = first extractor, second slot = second extractor” (order is defined by `extractor_order` at top level, which matches `result_multi.json`).

```json
{
  "schema_version": "multi_llm_v1",
  "paper_id": "0d5ffdaf23a655e1eff80bc8b6b4978067de4d5b",
  "extractor_order": ["claude-sonnet", "gpt-4o", "gemini-pro", "llama-3"],
  "materials": [
    {
      "material_name": "Fe phase nanosphere/conical CNT bundle nanostructures",
      "human_recipe": {
        "target_compound": "...",
        "target_compound_type": "...",
        "synthesis_method": "...",
        "starting_materials": [...],
        "steps": [...],
        "equipment": [...],
        "notes": "..."
      },
      "evaluations": [
        { "evaluation": { "reasoning": "...", "scores": {...}, "confidence_level": "...", ... } },
        { "evaluation": { "..." } },
        { "evaluation": { "..." } },
        { "evaluation": { "..." } }
      ]
    }
  ]
}
```

- **`extractor_order`**: Same ordered list as in `result_multi.json`. **Evaluations[0]** = score for extractor_order[0], **evaluations[1]** = score for extractor_order[1], etc. Annotators can ignore this and just fill the four boxes in order; scripts use it to align with the machine file.
- **`human_recipe`** (required): The **one** recipe for this material, written by the annotator. Same structure as the `synthesis` block in current single-LLM `result_human.json` (GeneralSynthesisOntology). No requirement to copy the paper verbatim — it is “your recipe” for this material, used as your reference when scoring.
- **`evaluations`**: **Array of 4** evaluation objects (not keyed by id). Each element has one `evaluation` (GeneralSynthesisEvaluation). **Position = extractor**: first item = score for first LLM’s extraction, second = score for second LLM’s extraction, etc.

**Why an array instead of keyed object:** Annotators can think “Score 1, Score 2, Score 3, Score 4” without matching “claude-sonnet” to the first box. Fewer mistakes and a simpler mental model.

---

## 5. Annotator workflow (minimal and clear)

1. **Open** `result_human.json`. For each material you see: **human_recipe** (one block) and **evaluations** (list of 4 empty slots).
2. **For each material:**
   - **Step 1 — Your recipe:** Fill **`human_recipe`** with the synthesis you consider correct for this material (same structure as single-LLM). This is your reference; it does not need to match the paper word-for-word.
   - **Step 2 — Score the 4 extractions:** Open `result_multi.json`, find this material’s **extractions**. The first extraction corresponds to **evaluations[0]**, the second to **evaluations[1]**, etc. Fill each of the 4 evaluation blocks (scores 1–5, reasoning, lists) in order.
3. Save `result_human.json`.

Mental model: **“My recipe, then Score 1, Score 2, Score 3, Score 4.”** No need to remember which model is which — order is fixed.

---

## 6. Generating the human template

Pipeline or a small script can generate `result_human.json` from `result_multi.json`:

- Copy `paper_id` and set **`extractor_order`** = `extractor_ids` from the machine file (so evaluations[i] aligns with extractor_ids[i]).
- For each material, set **`human_recipe`** to `null` or an empty GeneralSynthesisOntology template for annotators to fill.
- For each material, set **`evaluations`** to an **array of 4** empty evaluation objects (same schema as before; order = extractor_order).
- Annotators fill human_recipe first, then the four evaluation slots in order.

Example empty evaluation block (same as current schema):

```json
{
  "evaluation": {
    "reasoning": "",
    "scores": {
      "structural_completeness_score": null,
      "structural_completeness_reasoning": "",
      "material_extraction_score": null,
      "material_extraction_reasoning": "",
      "process_steps_score": null,
      "process_steps_reasoning": "",
      "equipment_extraction_score": null,
      "equipment_extraction_reasoning": "",
      "conditions_extraction_score": null,
      "conditions_extraction_reasoning": "",
      "semantic_accuracy_score": null,
      "semantic_accuracy_reasoning": "",
      "format_compliance_score": null,
      "format_compliance_reasoning": "",
      "overall_score": null,
      "overall_reasoning": ""
    },
    "confidence_level": null,
    "missing_information": [],
    "extraction_errors": [],
    "improvement_suggestions": []
  }
}
```

---

## 7. Backward compatibility

- **Single-LLM papers**: Keep using `result.json` (array of `{ material, synthesis, evaluation }`) and `result_human.json` (same array shape). No change for existing annotations.
- **Multi-LLM papers**: Use `result_multi.json` + `result_human.json` (multi schema with `schema_version: "multi_llm_v1"`).
- **Eval scripts**: Detect schema (single vs multi) per paper and:
  - Single: current logic (match by material, one human vs one LLM judge).
  - Multi: for each (material, extractor) pair, compare human evaluation to each of the 4 judge evaluations; aggregate to “which judge agrees most with humans” and “which extractor scores highest”.

---

## 8. Downstream analysis (what you can compute)

- **Judge agreement with humans**: For each judge J, compute agreement (e.g. Spearman, ICC) between J’s scores and human scores **across all (material, extractor) pairs** (or per extractor). Rank judges; pick the one with highest agreement.
- **Extractor ranking**: For each extractor E, average human score (e.g. overall_score) across materials. Optionally average using only the chosen judge’s scores. Pick the extractor with highest score.
- **Self-bias**: For each judge J, compare J’s score when evaluating J’s own extraction vs J’s score when evaluating others (e.g. mean score by judge–extractor same vs different). Test if “same” is systematically higher.

---

## 9. Summary

| Aspect | Single-LLM (current) | Multi-LLM (proposed) |
|--------|------------------------|----------------------|
| Machine file | `result.json` (array) | `result_multi.json` (object with materials + extractions + judge_evaluations) |
| Human file | `result_human.json` (array: material, synthesis, evaluation) | `result_human.json` (object: materials + human_recipe + evaluations[0..3]) |
| Human task | 1 human recipe + 1 evaluation per material | 1 human recipe + 4 evaluations per material (ordered list: Score 1..4) |
| Matching | By material name | By material_name + position in evaluations array (position ↔ extractor_order) |

Keeping **one human recipe** per material and **four score slots in order** (no model-name matching) makes the task intuitive; scripts use `extractor_order` to align with the machine file.

---

## 10. Related files

- **Annotator instructions**: [ANNOTATOR_INSTRUCTIONS_MULTI_LLM.md](ANNOTATOR_INSTRUCTIONS_MULTI_LLM.md) — step-by-step for annotators.
- **Example human template**: [example_result_human_multi_llm.json](example_result_human_multi_llm.json) — one material, four empty evaluation blocks.

---

## 11. Eval script changes (summary)

To support multi-LLM in your existing eval scripts (e.g. `compare_human_judge_scores_complete.py`):

1. **Detection**: For each paper, check for `result_multi.json` and, if present, whether `result_human.json` has `schema_version: "multi_llm_v1"` (or top-level `extractor_order`). If yes, treat as multi-LLM; else keep current single-LLM logic using `result.json` + `result_human.json`.

2. **Flatten to rows**: For multi-LLM, build one row per (paper_id, material_name, extractor_id):
   - Human scores from `result_human.json` → `materials[].evaluations[i].evaluation.scores` where `i` = index of extractor_id in `extractor_order`.
   - Judge scores from `result_multi.json` → `materials[].judge_evaluations[judge_id][extractor_id].evaluation.scores`.
   So you get a long table: (paper_id, material_id, extractor_id, human_overall_score, judge_a_score, …), etc.

3. **Judge agreement**: For each judge J, take all rows and compute Spearman/ICC between human overall (or per-criterion) and J’s score for that row. Rank judges by agreement; select best judge.

4. **Extractor ranking**: For each extractor E, average human overall_score across materials (and optionally average the chosen judge’s score for E’s extractions). Rank extractors; select best extractor.

5. **Self-bias**: For each judge J, compare mean(J’s score when extractor_id == J) vs mean(J’s score when extractor_id != J). Report difference or test.
