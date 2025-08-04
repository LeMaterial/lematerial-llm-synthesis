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
    Repairs the parsed dictionary output from the LLM to be more tolerant:
    - Maps out-of-schema actions (like "dissolve") to "mix" or "add".
    - Coerces blank/invalid booleans to None.
    - Converts things like "overnight" to a guessed numeric value or None.
    - Fixes degree symbols in temp_unit, and defaults empty strings.
    - Fixes starting_materials if necessary (see earlier code).
    """
    VALID_ACTIONS = {"add","mix","heat","reflux","age","filter","wash","dry","reduce","calcine"}
    # Map common out-of-vocab actions to closest allowed (expand as needed!)
    ACTION_MAP = {
        "dissolve": "mix",   # or "add", your science call!
        "stir": "mix",
        "precipitate": "add",
        "precipitation": "add",
        "combine": "mix",
        "cool": "age",
        # add more as needed
    }
    # Fix starting_materials if needed
    if "starting_materials" in parsed and parsed["starting_materials"]:
        if isinstance(parsed["starting_materials"][0], dict) and "name" in parsed["starting_materials"][0]:
            parsed["starting_materials"] = [
                m["name"] for m in parsed["starting_materials"] if "name" in m
            ]

    for step in parsed.get("steps", []):
        # Fix action case/mapping
        if "action" in step and isinstance(step["action"], str):
            action = step["action"].strip().lower()
            if action not in VALID_ACTIONS:
                action = ACTION_MAP.get(action, "add")  # lenient fallback
            step["action"] = action

        # Materials: heuristics as before (see previous code)

        for mat in step.get("materials", []):
            # Coerce empty or invalid role
            valid_roles = {"precursor", "support", "solvent", "additive", "reagent"}
            if "role" not in mat or not mat["role"] or mat["role"] not in valid_roles:
                name = mat.get("name", "").lower()
                if "water" in name:
                    mat["role"] = "solvent"
                else:
                    mat["role"] = "precursor"
            # Optionally, strip unit
            if "unit" in mat and isinstance(mat["unit"], str) and mat["unit"].strip() in {"°c", "°C"}:
                mat["unit"] = "C"

        # Fix condition fields
        if "conditions" in step:
            cond = step["conditions"]
            # Temperature unit to "C" if contains degree symbol
            if "temp_unit" in cond and isinstance(cond["temp_unit"], str):
                if "°" in cond["temp_unit"]:
                    cond["temp_unit"] = cond["temp_unit"].replace("°", "").replace(" ", "")

            # Convert booleans
            if "stirring" in cond:
                val = cond["stirring"]
                if isinstance(val, str):
                    low = val.strip().lower()
                    if low in ("true", "yes", "y", "stirred"):
                        cond["stirring"] = True
                    elif low in ("false", "no", "n", "not stirred"):
                        cond["stirring"] = False
                    else:
                        cond["stirring"] = None  # lenient, blank, or unknown

            # For durations like "overnight", "several hours" etc.
            if "duration" in cond:
                v = cond.get("duration")
                if isinstance(v, str):
                    if v.strip().lower() == "overnight":
                        cond["duration"] = 12  # or your preferred default
                        cond["time_unit"] = cond.get("time_unit") or "h"
                    elif v.strip().lower() in {"several hours","several hrs", "a few hours"}:
                        cond["duration"] = 3  # or your chosen value
                        cond["time_unit"] = cond.get("time_unit") or "h"
                    elif v == "":
                        cond["duration"] = None
                    else:
                        # Try to extract a number (with optional unit)
                        m = re.match(r"([0-9\.]+)\s*([a-zA-Z]*)", v)
                        if m:
                            cond["duration"] = float(m.group(1))
                            if m.group(2):
                                cond["time_unit"] = m.group(2)
                        else:
                            cond["duration"] = None

    return parsed




# ========== MAIN EXTRACTION FUNCTION ==========

def extract_synthesis(text: str) -> GeneralSynthesisOntology:
    full_prompt = build_prompt(text)
    predict = dspy.Predict(RawPromptToJson)
    result = predict(prompt=full_prompt)
    output = result.raw_json
    if "<|output|>" in output:
        output = output.split("<|output|>", 1)[-1].strip()

    # Try: JSON parse + ontology repair
    try:
        parsed = json.loads(output)
    except Exception:
        # Fallback: try to repair JSON, then parse
        try:
            repaired_json = repair_json(output)
            parsed = json.loads(repaired_json)
        except Exception as e2:
            print("Raw model output was:\n", output)
            raise ValueError("Failed to parse model output as JSON (even after repair).") from e2

    # Always apply ontology repair
    parsed = repair_for_ontology(parsed)

    # Try Pydantic parse
    try:
        return GeneralSynthesisOntology(**parsed)
    except Exception as e3:
        # (Optional) Print details of why Pydantic failed
        import traceback
        print("Pydantic validation failed after repair. Details:")
        traceback.print_exc()
        print("Raw (repaired + postprocessed) data:", json.dumps(parsed, indent=2))
        raise ValueError("Failed to validate as GeneralSynthesisOntology") from e3
        



# ========== TEST DRIVER ==========

if __name__ == "__main__":

    sample_text = ("""The synthesis and characterization of the mesoporous carbon nitride photocatalyst (mp-CN), the chemicals for the ex situ loading of Pt on to this material, the chemicals for the photocatalytic reaction, the setup and procedure for the ex situ method for co-catalyst loading (thermo-destabilization of microemulsions), and the setup for photocatalytic activity measurements on the lab scale are described in Schröder et al.18 This publication is open access and available to anyone.
Drop coating of Pt@mp-CN on a metal substrate - For the immobilization of the photocatalyst, stainless-steel plates with the following dimensions were used: 3.5 cm×3.5 cm×0.25 cm for the laboratory reactor and 30 cm×28 cm×0.1 cm for the demonstration reactor. In the drop-coating procedure, the following chemicals were used without further purification: Nafion (Nafion 117 solution 5 %, Sigma–Aldrich) as polymeric binder and ethanol (absolute, VWR Chemicals) as solvent. The Pt@mp-CN was dispersed in a mixture of ethanol and Nafion (9:1) and the steel plates were heated to 80 °C. Afterwards, the suspension (16 μL cm−2) was homogeneously distributed on the surface of the substrate and dried until all solvent was completely evaporated. Finally, the coated plate was dried (80 °C) under reduced pressure for 24 h.""")

    # sample_text = ("Summarizing,  we  have  uncovered  a  new  type  of  responsive  behaviour  in  open frameworks  involving interconversion between  meaningfully-different disordered states.  It  is  the  specific  type  of  symmetry  lowering  associated  with  stair-shaped framework  components  such  as  2,6-ndc,  that  when  combined  with  the  underlying square  lattice  of  the  DUT-8(Ni)  structure  leads  to  an  extensive  configurational landscape.  Host-guest  interactions  perturb  the  energetics  of  this  landscape  such that the system responds adaptively to guest adsorption. Repeated  solvent exchange demonstrates reversible switching between distinct disordered states. This new  type of framework  disorder-disorder switchability demonstrates a quasicontinuous dynamic transformation from one state into another. Each guest species encodes a complex microdomain structure (tiling) in the framework crystal recording the exposure history and molecular information in the material. The complex disorder pattern stores this type of information through the crystalline framework architecture. 48 Given the multitude of non-linear linkers employed for the design of MOFs,  COFs  and  more  recently  2D  materials  we  conceive  disorder-disorder switchability to be  of wider  importance  for  framework  materials  than  hitherto expected. Moreover, the control of disordered configurations may pave a new way to encode  open  frameworks  with  complex  information  and  consequently  distinct implications for their physical characteristics such as optical, magnetic and micromechanics. 48 In  particular  the  activation  energy  and  transformation  rate  of adaptive  pore  closing  and  opening  phenomena  are  expected  to  become  history dependent with important implications for the application of switchable MOFs in gas separation, storage and sensing devices.")
    
    # sample_text = (
    #     "A solution of 0.5 g of copper(II) nitrate trihydrate was dissolved in 10 mL of distilled water , then heated to 80 °C for 2 hours under a nitrogen atmosphere . The resulting blue precipitate was filtered and dried at 60 °C overnight ."
    # )

    # sample_text = ("A carefully chosen set of four different bis-monodentate ligands ( A B C , , and D ), all carrying the same kind of pyridyl donor groups, assemble with Pd II cations to a single nanoscopic cage [Pd 2 ABCD ] 4+ in  a  non-statistical fashion. Key for achieving this high degree of integrative self-sorting  is  the  combination  of  ligand  shape  complementarity,  balance  of  strain  and interligand C-H ⋯ π interactions. We demonstrate the modular replacement of ligands, even allowing to control the ligand order around the metal nodes (as in [Pd 2 ABD C 4 ] 4+ ). This paves the way for further derivatization to embed functionality. As the assembly proceeds under full thermodynamic control, formed products are robust and produced in a reproducible manner, following  several  alternative  routes  (directly  from  ligands  and  Pd II ,  by  mixing  homoleptic assemblies or stepwise via two- and three-component cages). Along these routes, more than 15 new  heteroleptic  cages  were  characterized,  including  nine  X-ray  structures.  Just  recently, rational self-assembly strategies push the field of supramolecular architecture to vastly increase complexity beyond homoleptic structures (one type of ligand per assembly) and bring multiple components  together  in  an  integrative  way.  With  the  presented  methodology,  we  have maximized the degree of ligand differentiation within the Pd II -cage family with the lowest nuclearity, allowing to selectively form one out of 55 possible products. As a general concept, this  will  allow  for  the  modular  development  of  multifunctional  assemblies  in  which  the interplay of different components leads to emerging properties, attractive for applications in selective recognition, cooperative catalysis and materials science.")
    
    ontology = extract_synthesis(sample_text)
    print(ontology.model_dump_json(indent=2))
