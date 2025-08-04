import json
import re
from typing import Literal
from pydantic import BaseModel, Field
import dspy
from dspy.adapters import TwoStepAdapter
from json_repair import repair_json

# ========== ONTOLOGY ==========

class Material(BaseModel):
    vendor: str | None = None
    name: str
    amount: float | None = None
    unit: str | None = None
    role: Literal["precursor", "support", "solvent", "additive", "reagent"]
    stoichiometry: str | None = None

class Conditions(BaseModel):
    temperature: float | None = None
    temp_unit: str | None = None
    duration: float | None = None
    time_unit: str | None = None
    atmosphere: str | None = None
    stirring: bool | None = None

class ProcessStep(BaseModel):
    action: Literal[
        "add", "mix", "heat", "reflux", "age", "filter", "wash", "dry", "reduce", "calcine"
    ]
    description: str | None = None
    materials: list[Material]
    conditions: Conditions | None = None

class GeneralSynthesisOntology(BaseModel):
    target_compound: str
    starting_materials: list[str]
    steps: list[ProcessStep]
    notes: str | None = None

# ========== STRICTER PROMPT ==========

FEW_SHOT_EXAMPLE = '''
### Example:
Template:
{
    "target_compound": "Li3(FeO3)2",
    "starting_materials": ["Lithium carbonate", "Iron(III) oxide"],
    "steps": [
        {
            "action": "mix",
            "description": "",
            "materials": [
                {
                    "vendor": null,
                    "name": "Lithium carbonate",
                    "amount": 1.0,
                    "unit": "g",
                    "role": "precursor",
                    "stoichiometry": null
                },
                {
                    "vendor": null,
                    "name": "Iron(III) oxide",
                    "amount": 2.0,
                    "unit": "g",
                    "role": "precursor",
                    "stoichiometry": null
                }
            ],
            "conditions": {
                "temperature": null,
                "temp_unit": null,
                "duration": null,
                "time_unit": null,
                "atmosphere": null,
                "stirring": null
            }
        },
        {
            "action": "heat",
            "description": "Calcine the mixture at 750C for 12 h in air.",
            "materials": [],
            "conditions": {
                "temperature": 750,
                "temp_unit": "C",
                "duration": 12,
                "time_unit": "h",
                "atmosphere": "air",
                "stirring": null
            }
        }
    ],
    "notes": null
}

Text:
"1g lithium carbonate and 2g iron(III) oxide were mixed thoroughly. The mixture was calcined at 750C for 12h in air to obtain Li3(FeO3)2."
'''

TEMPLATE = '''{
  "target_compound": "",
  "starting_materials": [],
  "steps": [
    {
      "action": "",
      "description": "",
      "materials": [],
      "conditions": {
        "temperature": null,
        "temp_unit": null,
        "duration": null,
        "time_unit": null,
        "atmosphere": null,
        "stirring": null
      }
    }
  ],
  "notes": null
}'''

# Prompt instructions
STRICT_INSTRUCTIONS = """
Respond ONLY with a minified valid JSON object matching the schema below.
- All numeric fields (amount, temperature, duration) MUST be numbers, NOT strings.
- All units (unit, temp_unit, time_unit) MUST be separate strings, NOT included in numbers.
- "materials" MUST be a list of objects with fields: name, amount, unit, vendor, role, stoichiometry.
- Use ONLY these exact actions (lowercase): add, mix, heat, reflux, age, filter, wash, dry, reduce, calcine.
- If a value is missing, use null. For booleans, use true/false/null (not strings!).
- Do not add any comments or explanation.
- Your output must be valid JSON and only valid JSON.

SCHEMA:
{template}

{fewshot}
"""

def build_prompt(text: str, template: str = TEMPLATE) -> str:
    instructions = STRICT_INSTRUCTIONS.format(
        template=json.dumps(json.loads(template), indent=4),
        fewshot=FEW_SHOT_EXAMPLE,
    )
    return f"""<|input|>
{instructions}
### Text:
{text}

<|output|>"""

# ========== DSPy SIGNATURE AND LLM SETUP ==========

class RawPromptToJson(dspy.Signature):
    prompt: str = dspy.InputField(desc="Prompt with NuExtract template, guidance, and <|input|>/<|output|> tokens.")
    raw_json: str = dspy.OutputField(desc="JSON block.")

lm = dspy.LM(
    'openai//scratch16/mshiel10/mzaki4/cache/models--numind--NuExtract-v1.5/snapshots/a7a4e41090a1c5aa95fdebab4c859d7111d628c0',
    api_base='http://localhost:8000/v1',
    api_key='EMPTY',   # or your key if needed
    temperature=0.7,
    max_tokens=20_000,
)
dspy.settings.configure(lm=lm, adapter=TwoStepAdapter(lm))

# ========== REPAIR AND POSTPROCESS ==========

def repair_for_ontology(parsed):
    """
    Repairs the parsed dictionary output from the model to match the requirements
    of the GeneralSynthesisOntology Pydantic schema.

    - Converts starting_materials list of dicts to list of strings (extracting 'name').
    - Ensures all material roles are set to a valid Literal.
      Defaults to 'solvent' for water and 'precursor' otherwise.

    Args:
        parsed (dict): The JSON-like dictionary output from the LLM.

    Returns:
        dict: The repaired, schema-compliant dictionary.
    """
    # --- Fix starting_materials: convert list of dicts to list of strings ---
    if "starting_materials" in parsed and parsed["starting_materials"]:
        if isinstance(parsed["starting_materials"][0], dict) and "name" in parsed["starting_materials"][0]:
            parsed["starting_materials"] = [
                m["name"] for m in parsed["starting_materials"] if "name" in m
            ]

    # --- Fix material roles in each process step ---
    for step in parsed.get("steps", []):
        for mat in step.get("materials", []):
            valid_roles = {"precursor", "support", "solvent", "additive", "reagent"}
            # If 'role' is missing or invalid or blank, set default
            if "role" not in mat or not mat["role"] or mat["role"] not in valid_roles:
                # Heuristic: water usually gets "solvent", everything else "precursor"
                name = mat.get("name", "").lower()
                if "water" in name:
                    mat["role"] = "solvent"
                else:
                    mat["role"] = "precursor"

    return parsed



# ========== MAIN EXTRACTION FUNCTION ==========

def extract_synthesis(text: str) -> GeneralSynthesisOntology:
    full_prompt = build_prompt(text)
    predict = dspy.Predict(RawPromptToJson)
    result = predict(prompt=full_prompt)
    output = result.raw_json
    if "<|output|>" in output:
        output = output.split("<|output|>", 1)[-1].strip()
    try:
        parsed = json.loads(output)
        parsed = repair_for_ontology(parsed)
        return GeneralSynthesisOntology(**parsed)
    except Exception:
        try:
            repaired_json = repair_json(output)
            parsed = json.loads(repaired_json)
            parsed = repair_for_ontology(parsed)
            return GeneralSynthesisOntology(**parsed)
        except Exception as e2:
            print("Raw model output was:\n", output)
            raise ValueError("Failed to parse (even after repair and postprocessing)") from e2


# ========== TEST DRIVER ==========

if __name__ == "__main__":

    sample_text = ("""The synthesis and characterization of the mesoporous carbon nitride photocatalyst (mp-CN), the chemicals for the ex situ loading of Pt on to this material, the chemicals for the photocatalytic reaction, the setup and procedure for the ex situ method for co-catalyst loading (thermo-destabilization of microemulsions), and the setup for photocatalytic activity measurements on the lab scale are described in Schröder et al.18 This publication is open access and available to anyone.
Drop coating of Pt@mp-CN on a metal substrate - For the immobilization of the photocatalyst, stainless-steel plates with the following dimensions were used: 3.5 cm×3.5 cm×0.25 cm for the laboratory reactor and 30 cm×28 cm×0.1 cm for the demonstration reactor. In the drop-coating procedure, the following chemicals were used without further purification: Nafion (Nafion 117 solution 5 %, Sigma–Aldrich) as polymeric binder and ethanol (absolute, VWR Chemicals) as solvent. The Pt@mp-CN was dispersed in a mixture of ethanol and Nafion (9:1) and the steel plates were heated to 80 °C. Afterwards, the suspension (16 μL cm−2) was homogeneously distributed on the surface of the substrate and dried until all solvent was completely evaporated. Finally, the coated plate was dried (80 °C) under reduced pressure for 24 h.""")

    # sample_text = (
    #     "Mix 1g of Nickel Nitrate with 2g of Cobalt Nitrate in 100ml Deionized Water "
    #     "at 80C for 2h under an N2 atmosphere. Filter and dry at 120C."
    # )
    # sample_text = (
    #     "A solution of 0.5 g of copper(II) nitrate trihydrate was dissolved in 10 mL of distilled water , then heated to 80 °C for 2 hours under a nitrogen atmosphere . The resulting blue precipitate was filtered and dried at 60 °C overnight ."
    # )
    ontology = extract_synthesis(sample_text)
    print(ontology.model_dump_json(indent=2))
