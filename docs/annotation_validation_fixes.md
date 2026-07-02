# Annotation validation — 2 files to fix

Before anything: **pull the latest `fix/recipe--annotations` branch.**

```bash
git checkout fix/recipe--annotations
git pull --ff-only
```

Then reproduce the errors:

```bash
uv run examples/scripts/data_curation/validate_result_human_schema.py --annotations-dir annotations
```

Current output:

```
FAIL  annotations/1df04f9e3f942b30d5e1c2bd1ab9cc3a79c23f13/result_human.json
        invalid JSON: Expecting ',' delimiter: line 887 column 2 (char 30145)

FAIL  annotations/f2f0828a5de4a3262edc73876809a9fe03ed6ff5/result_human.json
        [[C18C1im][NTf2]] <root>: unknown field 'evaluations'
        [[C18C1im][NTf2]] steps[4]: unknown field 'notes'
```

---

## 1. `1df04f9e…` — broken JSON

**Issue:** the second material (`10CeGd`) is wrapped inside a stray `"materials": [ {` that is never closed. The unbalanced bracket carries to the end of the file, which is why the parser only complains at the last line (887).

Lines 448–451 currently:

```
448    {
449    "materials": [      <- stray, delete
450    {                   <- stray, delete
451      "material_name": "10CeGd",
```

**Verified stray, not an unfilled slot:** `10CeGd` is a complete recipe (same `coprecipitation` method as `5CeGd`, 4 starting materials, 10 steps, 4 evaluations), and `5CeGd`/`10CeGd` are the only materials referenced anywhere in the file. The `"materials": [` is an accidental nested copy of the top-level key, not a placeholder for another recipe — so nothing was meant to be filled in here.

**Please confirm before fixing:** can you verify there was no intention to add a separate recipe under that `"materials": [` wrapper, and that `10CeGd` is the intended second (and final) material? Our read is that it's a stray paste artifact, but you annotated this — please sanity-check against the paper.

**Fix (once confirmed):** delete lines 449 and 450 so the material object opens directly with `"material_name"`. After this the file parses and contains 2 materials (`5CeGd`, `10CeGd`).

```bash
sed -i '' '449,450d' annotations/1df04f9e3f942b30d5e1c2bd1ab9cc3a79c23f13/result_human.json
```

---

## 2. `f2f0828a…` — mis-nested structure

This file parses as JSON but the structure is wrong. It should have **4 materials**, but the 4th (`[C18C1im][Cl]`) is swallowed inside material 3's (`[C18C1im][NTf2]`) `evaluations`. The two reported errors are symptoms:

- `<root>: unknown field 'evaluations'` — the material-level `evaluations` was nested *inside* `human_recipe` (and double-wrapped). That wrapper holds material 3's 4 evaluations **plus the entire `[C18C1im][Cl]` material object** plus that 4th material's evaluations.
- `steps[4]: unknown field 'notes'` — the recipe-level `human_recipe.notes` string was dropped into the 5th step instead.

**Fix (manual reconstruction — not a one-liner):**
1. Move `steps[4].notes` → `human_recipe.notes` on material 3.
2. Un-nest: set material 3's `evaluations` to the first 4 evaluation entries from the wrapper; remove `human_recipe.evaluations`.
3. Promote the swallowed `[C18C1im][Cl]` object to a 4th top-level material, with its `evaluations` taken from the same wrapper.

---

## Verify

Re-run until it reports **0 issues**:

```bash
uv run examples/scripts/data_curation/validate_result_human_schema.py --annotations-dir annotations
```
