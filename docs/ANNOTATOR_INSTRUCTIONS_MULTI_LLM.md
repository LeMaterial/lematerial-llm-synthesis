# Annotator instructions: Multi-LLM evaluation

When a paper has **multiple LLM extractions** (e.g. 4 different models each extracted a recipe), you do two things per material: **write your recipe** for that material, then **score each of the 4 extractions** in order. We use this to see which models extract and judge best.

---

## What you will see

- **`result_multi.json`** (read-only): Four extracted recipes per material (Extraction 1, 2, 3, 4). You don’t edit this file.
- **`result_human.json`** (you fill this): For each material — **your recipe** (one block) and **four score blocks** (Score 1, Score 2, Score 3, Score 4) in that order.

---

## Step-by-step (per material)

1. **Open `result_human.json`** and find the material. You’ll see:
   - **Your recipe** — one block to fill first.
   - **Evaluations** — a list of 4 slots (Score 1, Score 2, Score 3, Score 4).

2. **Fill “Your recipe”.**  
   In **`human_recipe`**, write the synthesis you consider correct for this material. Use the same structure as in the single-LLM task (target_compound, starting_materials, steps, equipment, notes, etc.). This is **your** reference recipe; it doesn’t need to match the paper word-for-word.

3. **Score the 4 LLM extractions.**  
   Open **`result_multi.json`**, find this material under **`extractions`**. There are 4 entries in a fixed order:
   - **Extraction 1** → fill **Score 1** (first evaluation slot) in `result_human.json`
   - **Extraction 2** → fill **Score 2** (second slot)
   - **Extraction 3** → fill **Score 3** (third slot)
   - **Extraction 4** → fill **Score 4** (fourth slot)

   You don’t need to remember which model is which — **position is enough**: first extraction = first score, etc.

4. **For each of the 4 score blocks**, fill the same fields as in the single-LLM task:
   - **reasoning**: Short overall comment.
   - **scores**: Same 1–5 scale (structural_completeness, material_extraction, process_steps, equipment_extraction, conditions_extraction, semantic_accuracy, format_compliance, overall_score) plus the short reasoning for each.
   - **confidence_level**: "low" / "medium" / "high"
   - **missing_information**, **extraction_errors**, **improvement_suggestions**: As needed.

5. **Save** `result_human.json`.

---

## Order and consistency

- **Your recipe** first, then **Score 1, Score 2, Score 3, Score 4** in that order. The order of the 4 extractions in `result_multi.json` matches the order of the 4 evaluation slots.
- Use the **same 1–5 scoring criteria** as in the single-LLM task. Only penalise for what’s wrong or missing in the extraction relative to what the paper actually says (don’t penalise for fields the paper doesn’t mention).

---

## Quick checklist per material

- [ ] Fill **your recipe** (`human_recipe`) for this material.
- [ ] Open `result_multi.json` → find this material’s extractions (1–4).
- [ ] Fill **Score 1** (evaluations[0]) for Extraction 1.
- [ ] Fill **Score 2** (evaluations[1]) for Extraction 2.
- [ ] Fill **Score 3** (evaluations[2]) for Extraction 3.
- [ ] Fill **Score 4** (evaluations[3]) for Extraction 4.
- [ ] Save `result_human.json`.

**Summary:** One human recipe + four scores in order. No need to match model names — first slot = first extractor, second = second, and so on.
