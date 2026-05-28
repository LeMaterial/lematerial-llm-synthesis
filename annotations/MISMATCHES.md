# Annotation Mismatches: `result_human.json` vs `old/result_human.json`

Comparison of the human-annotated recipe content (ignoring `evaluation`/`evaluations`,
which differ by design — OLD has one human evaluation, NEW has four LLM evaluations).

- **OLD format**: JSON list `[{material, synthesis, evaluation}, ...]`
- **NEW format**: JSON object `{schema_version, paper_id, ..., materials: [{material_name, human_recipe, evaluations[]}, ...]}`

The `synthesis` ↔ `human_recipe` payloads should match. Below are the cases where they don't.

---

## MAJOR — factual recipe mismatches (13 papers)

These contain real content drift: lost recipes, added/dropped materials, restructured steps, changed methods, or numeric/unit issues. Worth manual review.

### `90233593a9aa72b4bacfdeadc20050ae6d4b88e1` — possible numeric unit bug
- `synthesis_method`: OLD `"PVD"` → NEW `"magnetron sputtering"`
- `target_compound`: OLD `"CuMoTaWV High-Entropy Film"` → NEW `"CuMoTaWV"`
- Starting-material amounts (`0.2 mol%`) and units dropped to `null` in NEW
- `starting_materials`: NEW adds `"Si3N4 milling balls"`
- Steps reorganized 5 → 4 (NEW merges OLD steps 1+2; explicit `anneal` step folded into a `heat`)
- **Step pressure: OLD `1.16 Pa` vs NEW `0.00116 mPa` — off by ~1000×, likely a unit-conversion bug**
- Top-level equipment: OLD `[]` → NEW lists ball mill, SPS, sputtering system

### `9a889c1a671fd3cae48285eaa95069d189d02fe3` — recipes wiped
All 4 materials (Pt/Pd/Au/Ag-OLC) have **every recipe field nulled** in NEW:
`target_compound=null`, `synthesis_method=null`, `starting_materials=[]`, `steps=[]`, `equipment=[]`, `notes=null`. OLD had a full 2-step recipe (anneal at 1500 °C, then mix/hydrothermal at 200 °C) with 4 starting materials per entry.

### `2883daff26f16a13134a26ca5d366549a14fcc9c` — recipes wiped
OLD had 3 fully-populated graphene recipes (Graphene Sheets — electrochemical deposition, 6 steps; Graphene Quantum Dots — hydrothermal, 9 steps; Graphene Quantum Dots — electrochemical deposition, 4 steps). NEW collapses to a single `"No materials synthesized"` empty entry.

### `62b4ce6b486c987262b0ff80` — recipes wiped, materials added
- OLD: 1 material `"Au-Sn plasmonic alloy nanoparticles"` with full recipe (60 °C, SnCl4 + 15 nm citrate-capped Au seeds, 1 reduce step)
- NEW: 5 materials (Ag-Sn, Au-Bi, Au-In, Au-Ga, Au-Sn nanoparticles) **all with empty recipes** (`target_compound=null`, `synthesis_method=null`, no steps/materials/equipment)

### `1df04f9e3f942b30d5e1c2bd1ab9cc3a79c23f13` — Material 0 wiped
- Materials 1 (`5CeGd`) and 2 (`10CeGd`): identical
- Material 0: OLD `"ceria-doped zirconia ceramics"` had a full coprecipitation recipe (3 starting materials: zirconium oxychloride, cerium nitrate, ammonia solution; 3 steps: precipitate/dry/calcine). NEW has `target_compound=null`, `synthesis_method=null`, `starting_materials=[]`, `steps=[]`.

### `0d5ffdaf23a655e1eff80bc8b6b4978067de4d5b` — material lost
- OLD has 2 materials, NEW has 1
- The `"CNT array"` entry (CVD synthesis with xylene/ferrocene starting materials, 2 steps including 875 °C heat at 30 s) is **missing entirely** from NEW

### `1705.03436` — multiple materials lost / collapsed
- OLD has 11 material entries; NEW has 8
- Three rare-earth scandate entries (`DyScO3` annealing-only, `DyScO3` annealing+etching, `GdScO3`, `NdScO3`) are collapsed in NEW into a single placeholder `"REScO3"` with `target_compound=null`, all empty fields
- OLD's three SrTiO3 entries (001, 110, 111 surfaces) reduced in NEW to only the (111) surface; the 001 BHF etch + 1000 °C anneal recipe and the 110 anneal recipe are **missing**

### `2404.08872` — material added
- OLD has 1 material (`MOS2`); NEW has 2 (`MOS2` + `MOS2-rGO`)
- Shared `MOS2` matches (5 steps, equipment); `target_compound_type` dropped from `"two-dimensional materials"` → `null`
- New `MOS2-rGO`: full 7-step hydrothermal recipe with 4 starting materials (incl. graphene oxide) — entirely absent from OLD

### `2306.14755` — content rewritten
- Material renamed `ErTe3` → `ErTe3 single crystals` (cosmetic)
- `target_compound_type`: OLD `"semiconductors & electronic"` → NEW `"emerging & quantum materials"` (different category)
- OLD had empty `starting_materials` and `steps` (procedure only in `notes`); NEW adds 2 starting materials (`Er`, `Te`) and 1 `crystallize` step inferred from the notes
- `notes` text rewritten

### `cond-mat.0603598` — content rewritten
- `material_name`: OLD `"LaAlO3"` → NEW `"LaAlO3/SrTiO3"`
- `target_compound`: OLD `"LaAlO3"` → NEW `"LaAlO3/SrTiO3 heterointerface"`
- `target_compound_type`: OLD `"semiconductors & electronic"` → NEW `"two-dimensional materials"`
- Step count 4 → 5 with reorganization (added etching step 1, deposition steps 2–3, cool merging quench+cool, plus a new oxidation step 5 with ~6000 L oxygen at 150 °C)
- NEW adds equipment (RHEED, rectangular mask, variable attenuator) not in OLD's top-level list
- Step 3 conditions: OLD single `pressure: 1e-6` → NEW non-schema field `pressures: [1e-6, 1e-5]`; OLD step 4 `pressure: 0.00133 Pa` omitted in NEW
- NEW introduces non-schema step-level `notes` field and uses `conditions: null` in 3 steps

### `673b3fdd7be152b1d07c21f1` — name change + extra step + new equipment
- All four `"Oxidized CNS"` entries renamed to `"Carboxylate-functionalized cellulose nanospheres"` (in both `material_name` and `target_compound`)
- NEW added an extra step 5 (`"dry"` — "Lyophilizing the dialyzed water to obtain the product.") to all four syntheses (NEW=5 steps vs OLD=4 steps)
- NEW adds `dialysis membrane` equipment to step 4 and to top-level equipment list
- Last entry's step 2 action mistyped as `"ball mixing"` in NEW vs `"ball milling"` in OLD
- NEW drops `stirring: "yes"` for synthesis 2 (kept in OLD for ball-milled cellulose)

### `73c6aeebd5877d2eb17d4961577d98216d503e6f` — step count change
- OLD had 6 steps (with three duplicate `"wash"` steps that look like a bug); NEW collapses to 4 (single wash) — procedural cleanup
- Starting materials: NEW has `P. frutescens extract aqueous solution` with `amount: null`; OLD used non-standard key `quantity: 1.0`
- Step 1 materials: NEW flattens `"AgNO3 (0.5 mM)"` into name with structured fields nulled; OLD has structured `name: "AgNO3", amount: 0.5, unit: "mM"`

### `64b40972b605c6803bd37ab4` — synthesis_method changed
- Both materials (WFe2Ni-red, WFe2Ni-ox): `synthesis_method` `"simulation"` (OLD) → `"other"` (NEW)
- All other content (starting materials, single step, conditions, notes) is identical

### `65e0464ce9ebbb4db98ae397` — type cleared, fields flattened
- `Fmoc-LIVKHH-NH2`: `target_compound_type` `"biomaterials & biological"` → `null`; step materials flattened into name strings (e.g., `"Fmoc-rink amide AM resin (0.78 mmol/g) [vendor: Merck]"`) instead of OLD's structured `{vendor: "Merck", amount: 0.78, unit: "mmol/g"}`
- `metallo-hydrogels`: `target_compound_type` `"hybrid & organic-inorganic"` → `null`; same flattening (e.g., `"Zn(OAc)2 (0.5 equiv)"` with `amount: null`)
- `Ac-LIVKHH-NH2`: identical

---

## MINOR — cosmetic / schema-format only (16 papers)

Recipes match factually; differences are limited to common patterns (see below).

`0307fa0472a682ab559ac343038ed0cfeb8fe815`, `1602.02498`, `1605.04038`, `1706.00484`,
`1902.03049`, `2212.12506`, `2502.03121`, `3325ac6dfb049a5efdaad7f876c1d51b17be0158`,
`3b87159630bb0581024897961bb5fc922fc3db19`, `60c74548469df43eacf434a6`,
`61b80eb702d90d55416229c1`, `c47e0cbc8b6feb8d28c3d9c1c29f98772ede6c27` (also dedups
duplicate entries: 14 → 12), `ccc7c5d70ae3ca3f9e975d0dc3b4d631586c1586`,
`cond-mat.0503432`, `cond-mat.0607131`, `f2f0828a5de4a3262edc73876809a9fe03ed6ff5`

### Common minor patterns
- `target_compound_type` set to `null` in NEW (OLD had a category)
- Step-level material structured fields (`vendor`/`amount`/`unit`/`purity`) flattened
  into the `name` string in NEW, with structured fields nulled
  (e.g., `"Te powder (0.96 g) [vendor: Sigma Aldrich, Sweden] [purity: 99.8%]"`)
- Atmosphere phrasing (`"argon"` → `"high purity argon"`)
- `stirring: true` dropped to `null` in NEW
- Equipment grouping (`name: "copper hearth", settings: "water-cooled"` →
  `name: "water-cooled copper hearth", settings: null`)

---

## IDENTICAL — recipes truly match (8 papers)

`1409.1070`, `1709.00477`, `62cff0f127b1e42fe039c25e`, `8c37fd10addf6d79f84ec2d5f4a8e5c6d6ef676f`,
`914dfcfe8762e189e9d7873090587458e7c86695`, `cond-mat.0510550`, `cond-mat.0602418`,
`cond-mat.9604170`

---

## Suggested priority order for review

1. **`90233593...`** — possible numeric unit bug in step pressure (1.16 Pa vs 0.00116 mPa). Investigate first.
2. **Recipes silently wiped**: `9a889c1a...`, `2883daff...`, `62b4ce6b...`, `1df04f9e...` — verify whether these were intentional rollbacks or data loss.
3. **Materials/recipes dropped**: `0d5ffdaf...`, `1705.03436` — check whether the missing entries should be restored.
4. **Content rewritten / restructured**: `2306.14755`, `2404.08872`, `cond-mat.0603598`, `673b3fdd...`, `73c6aeebd...`.
5. **Field-level changes**: `64b40972...`, `65e0464c...`.
