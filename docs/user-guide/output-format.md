# Understanding the Output Format

This page explains every field in the JSON files that LeMat-Synth produces.
You do not need to read any code to use this guide.

---

## Where are my result files?

After running any extraction script or `lemat-synth extract`, results are saved in a
folder structure like this:

```
results/
└── <paper-id>/
    ├── Fe2O3.json                    ← one file per synthesized material
    ├── Ni-Fe2O3.json
    ├── performance_mappings.json     ← plot-to-material links (full pipeline only)
    ├── linking_summary_llm.json      ← LLM quality evaluation of the links
    └── linking_summary_human.json    ← blank template for your own annotation
```

Each `<material>.json` file is the main result. The other files are only present when
you ran the pipeline with figure extraction enabled (`--with-performance`).

---

## The synthesis result file (`<material>.json`)

### Minimal example

```json
{
  "material": "3%Ru/CaO",
  "synthesis": {
    "target_compound": "3%Ru/CaO",
    "target_compound_type": "functional materials & catalysts",
    "synthesis_method": "wet impregnation",
    "starting_materials": [
      {
        "name": "RuCl3",
        "amount": 0.12,
        "unit": "g",
        "vendor": "Sigma-Aldrich",
        "purity": "99%"
      },
      {
        "name": "CaO",
        "amount": 1.0,
        "unit": "g",
        "vendor": null,
        "purity": null
      },
      {
        "name": "Deionized Water",
        "amount": 20.0,
        "unit": "mL",
        "vendor": null,
        "purity": null
      }
    ],
    "steps": [
      {
        "step_number": 1,
        "action": "dissolve",
        "description": "Dissolve RuCl3 in deionized water to form a precursor solution.",
        "materials": [
          {"name": "RuCl3", "amount": 0.12, "unit": "g", "vendor": null, "purity": null},
          {"name": "Deionized Water", "amount": 20.0, "unit": "mL", "vendor": null, "purity": null}
        ],
        "equipment": [{"name": "magnetic stirrer", "instrument_vendor": null, "settings": "room temperature"}],
        "conditions": {
          "temperature": null,
          "temp_unit": null,
          "duration": 30.0,
          "time_unit": "min",
          "atmosphere": "air",
          "stirring": true,
          "stirring_speed": null,
          "pressure": null,
          "pressure_unit": null,
          "ph": null
        }
      },
      {
        "step_number": 2,
        "action": "impregnate",
        "description": "Add CaO support to the precursor solution and stir.",
        "materials": [{"name": "CaO", "amount": 1.0, "unit": "g", "vendor": null, "purity": null}],
        "equipment": [],
        "conditions": {"temperature": null, "temp_unit": null, "duration": 2.0, "time_unit": "h",
                       "atmosphere": "air", "stirring": true, "stirring_speed": 300.0,
                       "pressure": null, "pressure_unit": null, "ph": null}
      },
      {
        "step_number": 3,
        "action": "dry",
        "description": "Dry the impregnated sample overnight.",
        "materials": [],
        "equipment": [{"name": "oven", "instrument_vendor": null, "settings": null}],
        "conditions": {"temperature": 110.0, "temp_unit": "C", "duration": 12.0,
                       "time_unit": "h", "atmosphere": "air", "stirring": false,
                       "stirring_speed": null, "pressure": null, "pressure_unit": null, "ph": null}
      },
      {
        "step_number": 4,
        "action": "calcine",
        "description": "Calcine the dried catalyst in air at 500 °C.",
        "materials": [],
        "equipment": [{"name": "tube furnace", "instrument_vendor": null, "settings": "5°C/min ramp"}],
        "conditions": {"temperature": 500.0, "temp_unit": "C", "duration": 4.0,
                       "time_unit": "h", "atmosphere": "air", "stirring": false,
                       "stirring_speed": null, "pressure": null, "pressure_unit": null, "ph": null}
      }
    ],
    "equipment": [
      {"name": "magnetic stirrer", "instrument_vendor": null, "settings": null},
      {"name": "oven", "instrument_vendor": null, "settings": null},
      {"name": "tube furnace", "instrument_vendor": null, "settings": "5°C/min ramp"}
    ],
    "notes": "Catalyst was reduced in H2 prior to activity measurement (not part of synthesis)."
  },
  "evaluation": {
    "structural_completeness_score": 4.5,
    "structural_completeness_reasoning": "All major fields populated; minor detail on drying oven vendor missing.",
    "material_extraction_score": 5.0,
    "material_extraction_reasoning": "Correct amounts, units, and purity values extracted.",
    "process_steps_score": 4.0,
    "process_steps_reasoning": "Steps in correct order; drying and calcination captured.",
    "equipment_extraction_score": 4.5,
    "equipment_extraction_reasoning": "Furnace and stirrer captured; oven brand missing.",
    "conditions_extraction_score": 5.0,
    "conditions_extraction_reasoning": "Temperatures, durations, and atmospheres all correct.",
    "semantic_accuracy_score": 5.0,
    "semantic_accuracy_reasoning": "All extracted information is faithful to the source.",
    "format_compliance_score": 5.0,
    "format_compliance_reasoning": "Schema fully respected.",
    "overall_score": 4.71
  },
  "performance": null
}
```

---

## Field-by-field explanation

### Top level

| Field | Type | Description |
|-------|------|-------------|
| `material` | string | The material name exactly as extracted from the paper |
| `synthesis` | object | The structured synthesis procedure (see below) |
| `evaluation` | object | Quality scores from the LLM judge (see below) |
| `performance` | object or null | Plot-linked performance data; `null` unless `--with-performance` was used |

---

### `synthesis` object

| Field | Type | Description |
|-------|------|-------------|
| `target_compound` | string | Chemical formula or name of the synthesized material |
| `target_compound_type` | string | One of 16 fixed categories (see table below) |
| `synthesis_method` | string | One of 35 fixed synthesis methods (see table below) |
| `starting_materials` | list | Reagents and precursors used (see `Material` below) |
| `steps` | list | Ordered synthesis steps (see `ProcessStep` below) |
| `equipment` | list | All equipment mentioned in the synthesis (see `Equipment` below) |
| `notes` | string or null | Anything relevant that did not fit the structured fields |

#### `target_compound_type` — allowed values

| Value | Examples |
|-------|---------|
| `metals & alloys` | Cu, Fe-Ni alloy, stainless steel |
| `ceramics & glasses` | Al₂O₃, SiO₂, BaTiO₃ |
| `polymers & soft matter` | PDMS, PET, hydrogel |
| `composites` | carbon fibre / epoxy, metal matrix |
| `semiconductors & electronic` | GaAs, Si, InP |
| `nanomaterials` | Au nanoparticles, TiO₂ nanorods |
| `two-dimensional materials` | graphene, MoS₂ monolayer |
| `framework & porous materials` | MOF, zeolite, COF |
| `biomaterials & biological` | hydroxyapatite, collagen scaffold |
| `liquid materials` | ionic liquid, solution |
| `hybrid & organic-inorganic` | perovskite, organosilica |
| `functional materials & catalysts` | Pt/Al₂O₃, zeolite catalyst |
| `energy & sustainability` | LiFePO₄, solar cell absorber |
| `smart & responsive materials` | shape-memory alloy, pH-responsive gel |
| `emerging & quantum materials` | topological insulator, qubit |
| `other` | anything not fitting above |

#### `synthesis_method` — common values

| Value | Brief description |
|-------|------------------|
| `sol-gel` | Hydrolysis/condensation of metal alkoxides |
| `hydrothermal` | Reaction in sealed autoclave with water at high T/P |
| `solvothermal` | Same but with non-aqueous solvent |
| `wet impregnation` | Soak support in precursor solution, dry, calcine |
| `incipient wetness impregnation` | Add just enough solution to fill pore volume |
| `precipitation` / `coprecipitation` | Precipitate from solution by pH or temperature change |
| `solid-state` | Mix powders and sinter/calcine |
| `CVD` | Chemical vapour deposition |
| `PVD` | Physical vapour deposition |
| `ball milling` | Mechanical grinding of powders |
| `electrochemical deposition` | Deposit material via applied current/potential |
| `combustion` | Auto-ignition of metal-nitrate/fuel mixture |
| `atomic layer deposition` | Alternating gas-phase precursor cycles |
| `arc melting & induction melting` | High-temperature melting under controlled atmosphere |
| `mechanochemical` | Solid-state reaction driven by mechanical energy |
| `other` | Method not in the list |

---

### `Material` object (inside `starting_materials` and `steps[].materials`)

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Chemical name or formula |
| `amount` | number or null | Numeric quantity only (e.g. `0.12`) |
| `unit` | string or null | Unit of amount (e.g. `"g"`, `"mL"`, `"mmol"`, `"wt%"`) |
| `vendor` | string or null | Supplier name if mentioned |
| `purity` | string or null | Purity if stated (e.g. `"99%"`, `"ACS grade"`) |

---

### `ProcessStep` object (inside `steps`)

| Field | Type | Description |
|-------|------|-------------|
| `step_number` | integer | Position in the sequence (starting from 1) |
| `action` | string | One of the controlled verbs (see below) |
| `description` | string or null | Free-text description of the step from the paper |
| `materials` | list of `Material` | Materials involved in this specific step |
| `equipment` | list of `Equipment` | Equipment used in this step |
| `conditions` | `Conditions` or null | Physical conditions for this step |

**Allowed `action` values:** `add`, `mix`, `heat`, `cool`, `reflux`, `age`, `filter`,
`wash`, `dry`, `reduce`, `calcine`, `dissolve`, `precipitate`, `centrifuge`, `sonicate`,
`anneal`, `ion exchange`, `impregnate`

---

### `Conditions` object (inside `steps[].conditions`)

| Field | Type | Description |
|-------|------|-------------|
| `temperature` | number or null | Numeric value only (e.g. `500.0`) |
| `temp_unit` | string or null | `"C"`, `"K"`, or `"F"` |
| `duration` | number or null | Numeric value only |
| `time_unit` | string or null | `"h"`, `"min"`, `"s"`, `"days"` |
| `pressure` | number or null | Numeric value only |
| `pressure_unit` | string or null | `"atm"`, `"bar"`, `"Pa"`, `"torr"`, `"psi"` |
| `atmosphere` | string or null | Gas phase (e.g. `"air"`, `"N2"`, `"H2"`, `"Ar"`) |
| `stirring` | boolean or null | Whether stirring was used |
| `stirring_speed` | number or null | Speed in rpm |
| `ph` | number or null | pH value |

A `null` value means the paper did not mention that condition — it does **not** mean
the condition was absent.

---

### `Equipment` object

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Instrument name (e.g. `"autoclave"`, `"tube furnace"`) |
| `instrument_vendor` | string or null | Manufacturer if mentioned |
| `settings` | string or null | Operating settings (e.g. `"heating rate 5°C/min"`) |

---

### `evaluation` object — quality scores

Each dimension is scored from **1** (poor) to **5** (excellent) by the LLM judge.
A score of `null` means the judge did not produce a score for that dimension.

| Field | What is being evaluated |
|-------|------------------------|
| `structural_completeness_score` | Are all schema fields populated where information exists? |
| `material_extraction_score` | Are names, amounts, units, and purities correct? |
| `process_steps_score` | Are steps in the right order and correctly classified? |
| `equipment_extraction_score` | Is all mentioned equipment captured? |
| `conditions_extraction_score` | Are temperatures, times, atmospheres, pressures correct? |
| `semantic_accuracy_score` | Is the meaning of each step faithfully preserved? |
| `format_compliance_score` | Does the output conform to the schema? |
| `overall_score` | Arithmetic mean of all above scores |

Each score comes with a `*_reasoning` field explaining the rationale.

> **Note on low scores:** A low score means the extraction is incomplete or inaccurate
> relative to the source paper. It does **not** mean the synthesis itself was poor.
> The judge follows the rule "absence is not an error" — it will not penalise for
> omitting information that was never in the paper.

---

## Performance data (`performance` field)

Only present when `--with-performance` is used. Contains plot-linked data for this material.

```json
"performance": {
  "material_name": "3%Ru/CaO",
  "plot_data": [
    {
      "plot_index": 0,
      "figure_reference": "Fig. 3a",
      "series_name": "3%Ru/CaO",
      "coordinates": [[200, 5.2], [250, 18.7], [300, 42.1], [350, 76.3], [400, 91.4]],
      "x_axis_label": "Temperature",
      "x_axis_unit": "°C",
      "y_axis_label": "CO conversion",
      "y_axis_unit": "%",
      "plot_title": "Catalytic activity vs. temperature",
      "confidence": "high"
    }
  ]
}
```

| Field | Description |
|-------|-------------|
| `plot_index` | Index of the figure in the paper (0-based) |
| `figure_reference` | Label as it appears in the paper (e.g. `"Fig. 3a"`) |
| `series_name` | Legend entry from the plot |
| `coordinates` | List of `[x, y]` pairs read from the plot |
| `x_axis_label` / `y_axis_label` | Axis labels |
| `x_axis_unit` / `y_axis_unit` | Axis units |
| `confidence` | Linking confidence: `"high"`, `"medium"`, or `"low"` |

---

## Reading results in Python

```python
import json
from pathlib import Path

result_file = Path("results/my_paper/Fe2O3.json")
data = json.loads(result_file.read_text())

print(data["material"])                        # "Fe2O3"
print(data["synthesis"]["synthesis_method"])   # "hydrothermal"
print(data["evaluation"]["overall_score"])     # 4.3

for step in data["synthesis"]["steps"]:
    print(f"Step {step['step_number']}: {step['action']} — {step['description']}")
```
