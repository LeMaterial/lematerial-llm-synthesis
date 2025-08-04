import json
from typing import Literal
from pydantic import BaseModel, Field
import dspy
from dspy.adapters import TwoStepAdapter

# ========== ONTOLOGY DEFINITIONS ==========
class Material(BaseModel):
    vendor: str | None = Field(
        None, description="Vendor of the material."
    )
    name: str = Field(
        ..., description="Name of the material."
    )
    amount: float = Field(
        ..., description="Amount of material used in the synthesis. Just the number, no unit."
    )
    unit: str = Field(
        ..., description="Unit of the amount. E.g. 'g', 'mol', 'wt%'."
    )
    role: Literal["precursor", "support", "solvent", "additive", "reagent"]
    stoichiometry: str | None = Field(
        None, description="Stoichiometry of the material in the synthesis."
    )

class Conditions(BaseModel):
    temperature: float | None = Field(None, description="Temperature of the synthesis.")
    temp_unit: str | None = Field(None, description="Unit of the temperature.")
    duration: float | None = Field(None, description="Duration of the synthesis.")
    time_unit: str | None = Field(None, description="Unit of the duration.")
    atmosphere: str | None = Field(None, description="Atmosphere of the synthesis.")
    stirring: bool | None = Field(None, description="Whether the synthesis is stirred.")

class ProcessStep(BaseModel):
    action: Literal[
        "add", "mix", "heat", "reflux", "age", "filter", "wash", "dry", "reduce", "calcine"
    ]
    description: str | None = Field(None, description="Description of the process step.")
    materials: list[Material] = Field(..., description="Materials used in the process step.")
    conditions: Conditions | None = Field(None, description="Conditions of the process step.")

class GeneralSynthesisOntology(BaseModel):
    target_compound: str = Field(..., description="Target compound composition.")
    starting_materials: list[str] = Field(..., description="Starting materials used in the synthesis.")
    steps: list[ProcessStep] = Field(..., description="Process steps of the synthesis.")
    notes: str | None = Field(None, description="Notes about the synthesis.")

# ========== PROMPT TEMPLATE ==========
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

def build_prompt(text: str, template: str = TEMPLATE) -> str:
    template_str = json.dumps(json.loads(template), indent=4)
    return f"""<|input|>
### Template:
{template_str}
### Text:
{text}

<|output|>"""

# ========== Dspy Signature ==========
class RawPromptToJson(dspy.Signature):
    prompt: str = dspy.InputField(desc="Prompt with <|input|>, <|output|>, schema, and synthesis text.")
    raw_json: str = dspy.OutputField(desc="The returned JSON block.")

# ========== DSPY LLM ==========

lm = dspy.LM(
    # Use LiteLLM-style path for vllm OpenAI backend
    'openai//scratch16/mshiel10/mzaki4/cache/models--numind--NuExtract-v1.5/snapshots/a7a4e41090a1c5aa95fdebab4c859d7111d628c0',
    api_base='http://localhost:8000/v1',  # Your vllm endpoint
    api_key='EMPTY',           # or set if your server requires
    temperature=0.7,
    max_tokens=20_000,
)

dspy.settings.configure(lm=lm, adapter=TwoStepAdapter(lm))

from json_repair import repair_json  # <-- Add this import

def extract_synthesis(text: str) -> GeneralSynthesisOntology:
    full_prompt = build_prompt(text)
    predict = dspy.Predict(RawPromptToJson)
    result = predict(prompt=full_prompt)
    output = result.raw_json
    if "<|output|>" in output:
        output = output.split("<|output|>", 1)[-1].strip()
    try:
        parsed = json.loads(output)
        return GeneralSynthesisOntology(**parsed)
    except Exception:
        # Fallback: attempt to repair malformed json and parse again!
        try:
            repaired_json = repair_json(output)
            parsed = json.loads(repaired_json)
            return GeneralSynthesisOntology(**parsed)
        except Exception as e2:
            print("Raw model output was:\n", output)
            raise ValueError("Failed to parse (even after repair) model output as JSON.") from e2


# ========== TEST DRIVER ==========
if __name__ == "__main__":
    sample_text = (
        "Mix 1g of Nickel Nitrate with 2g of Cobalt Nitrate in 100ml Deionized Water "
        "at 80C for 2h under an N2 atmosphere. Filter and dry at 120C."
    )
    ontology = extract_synthesis(sample_text)
    print(ontology.model_dump_json(indent=2))



# import dspy
# import json
# from typing import Literal
# from pydantic import BaseModel, Field
# from dspy.adapters import TwoStepAdapter

# # Ontology definitions
# class Material(BaseModel):
#     vendor: str | None = Field(None, description="Vendor of the material. E.g. 'Sinopharm Chemical Reagent Co. Ltd.'.")
#     name: str = Field(..., description="Name of the material. E.g. 'Nickel Nitrate', 'Cobalt Nitrate', 'Deionized Water', 'Ammonia Solution'.")
#     amount: float = Field(..., description="Amount of material used in the synthesis. Just the number, no unit.")
#     unit: str = Field(..., description="Unit of the amount. E.g. 'g', 'mol', 'wt%'.")
#     role: Literal["precursor", "support", "solvent", "additive", "reagent"]
#     stoichiometry: str | None = Field(None, description="Stoichiometry of the material in the synthesis. E.g. '1:1', '1:2', '2:1'.")

# class Conditions(BaseModel):
#     temperature: float | None = Field(None, description="Temperature of the synthesis. E.g. 100, 200, 300.")
#     temp_unit: str | None = Field(None, description="Unit of the temperature. E.g. 'C', 'K'.")
#     duration: float | None = Field(None, description="Duration of the synthesis. E.g. 1, 2, 3.")
#     time_unit: str | None = Field(None, description="Unit of the duration. E.g. 'h', 'min', 's'.")
#     atmosphere: str | None = Field(None, description="Atmosphere of the synthesis. E.g. 'air', 'N2', 'H2'.")
#     stirring: bool | None = Field(None, description="Whether the synthesis is stirred.")

# class ProcessStep(BaseModel):
#     action: Literal["add", "mix", "heat", "reflux", "age", "filter", "wash", "dry", "reduce", "calcine"]
#     description: str | None = Field(None, description="Description of the process step.")
#     materials: list[Material] = Field(..., description="Materials used in the process step.")
#     conditions: Conditions | None = Field(None, description="Conditions of the process step.")

# class GeneralSynthesisOntology(BaseModel):
#     target_compound: str = Field(..., description="Target compound composition.")
#     starting_materials: list[str] = Field(..., description="Starting materials used in the synthesis.")
#     steps: list[ProcessStep] = Field(..., description="Process steps of the synthesis.")
#     notes: str | None = Field(None, description="Notes about the synthesis.")

# # System prompt
# system_prompt = """You are LeMat-SynthP, a large-language model (transformer) that:
# - Reliably EXTRACTS and EVALUATES materials-synthesis procedures from scientific papers.
# - Outputs JSON objects that match the "GeneralSynthesisOntology" schema.

# Persona  
# - Helpful, concise, rigorous scientific assistant.  
# - Expert in chemistry (materials science, catalysis, nanostructures).  
# - Speaks in formal scientific English; uses real-world analogies sparingly to clarify concepts.

# Grounding rules  
# 1. Cite or quote the exact sentence/figure that supports every non-obvious claim.  
# 2. If evidence is missing or ambiguous, answer "INSUFFICIENT_EVIDENCE" instead of guessing.  
# 3. Do not invent data, quantities, or references.

# Extraction rules  
# - Always include reagents, exact quantities + units, atmosphere, temperature, time, pH, and supports (with adjectives: "doped", "functionalized", …).  
# - Expand abbreviations (e.g. "Al2O3" ⇢ "Aluminium oxide (Al2O3)").  
# - Capture catalyst supports even when cryptic (e.g. "CZY", "AC-FS").  
# - Preserve stoichiometric ratios verbatim (e.g. "Li3(FeO3)2").

# Safety & policy (inherited from Anthropic constitutional policy May-2025)  
# - Harmlessness: refuse instructions that facilitate illicit synthesis of hazardous materials.  
# - Honesty: state uncertainty and limitations clearly.  
# - Privacy: redact personal data.  
# - Copyright: transform, summarise, or quote ≤ 90 characters at a time.

# Response-format contract  
# ```json
# {
#   "target_compound": "<string>",
#   "materials": ["<string>", …],
#   "steps": [
#     {
#       "action": "<add|mix|heat|reflux|age|filter|wash|dry|reduce|calcine>",
#       "materials": ["<string>", …],
#       "conditions": {
#         "temperature": <float|null>,
#         "temp_unit": "<C|K|null>",
#         "duration": <float|null>,
#         "time_unit": "<h|min|s|null>",
#         "atmosphere": "<string|null>",
#         "stirring": <true|false|null>
#       }
#     }
#   ],
#   "notes": "<string|null>"
# }
# ```
# If the input figure/text is non-scientific (e.g. logo) respond exactly: `NON_SCIENTIFIC_FIGURE`.
# Do **not** output anything outside the specified JSON (or single token above) unless explicitly instructed. The output SHOULD ALWAYS be a JSON""" 



# # Use your local endpoint and model name
# lm = dspy.LM(
#     # Use LiteLLM naming: 'openai/<model-name>'
#     'openai//scratch16/mshiel10/mzaki4/cache/models--numind--NuExtract-v1.5/snapshots/a7a4e41090a1c5aa95fdebab4c859d7111d628c0',
#     api_base='http://localhost:8000/v1',
#     api_key='EMPTY',   # or your API key if needed
#     temperature=0.7,    # any other params you'd like
#     max_tokens=20_000,
#     system_prompt=system_prompt,
# )

# from dspy.adapters import TwoStepAdapter
# dspy.settings.configure(lm=lm, adapter=TwoStepAdapter(lm))

# # Define DSPy signature
# class SynthesisExtraction(dspy.Signature):
#     """Extract synthesis ontology from text."""
#     synthesis_paragraph: str = dspy.InputField(desc="The synthesis paragraph to extract from.")
#     ontology: GeneralSynthesisOntology = dspy.OutputField(desc="Extracted synthesis ontology in JSON format.")

# # Example usage function
# def extract_synthesis(text: str) -> GeneralSynthesisOntology:
#     predict = dspy.Predict(SynthesisExtraction)
#     result = predict(synthesis_paragraph=text)
#     try:
#         # Parse the output to ontology
#         parsed_json = json.loads(result.ontology)
#         return GeneralSynthesisOntology(**parsed_json)
#     except (json.JSONDecodeError, ValueError) as e:
#         raise ValueError(f"Failed to parse output: {result.ontology}") from e

# # Sample test (replace with actual text)
# if __name__ == "__main__":
#     sample_text = "Example synthesis paragraph: Mix 1g of Nickel Nitrate with 2g of Cobalt Nitrate in 100ml Deionized Water at 80C for 2h under N2 atmosphere."
#     ontology = extract_synthesis(sample_text)
#     print(ontology.model_dump_json(indent=2))
