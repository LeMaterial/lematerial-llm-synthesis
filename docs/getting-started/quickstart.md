# Quickstart

This page gets you from zero to your first extracted synthesis in under 10 minutes.

## Option 1 — Interactive notebook (recommended for beginners)

Open the quickstart notebook in Jupyter:

```bash
uv run jupyter lab examples/notebooks/00_quickstart.ipynb
```

The notebook walks you through each step with explanations, lets you paste or load
a paper, and saves the result as a JSON file.

---

## Option 2 — Command line (one paper)

```bash
# Make sure your .env file has GEMINI_API_KEY set
source .env

# Extract synthesis from one text file
lemat-synth extract my_paper.txt

# Extract from a PDF (text is extracted automatically with Docling)
lemat-synth extract my_paper.pdf

# Results are saved to results/<paper-name>/<material>.json
```

---

## Option 3 — Command line (batch)

```bash
lemat-synth batch /path/to/my_papers/ results/
```

Add `--max 5` to process only the first 5 papers as a test:

```bash
lemat-synth batch /path/to/my_papers/ results/ --max 5
```

---

## Option 4 — Python API (one paper, no config files)

```python
import json
import os
from dotenv import load_dotenv

load_dotenv()

from llm_synthesis.utils.dspy_utils import get_llm_from_name
from llm_synthesis.utils import configure_dspy
from llm_synthesis.transformers.material_extraction.dspy_extraction import (
    DspyTextExtractor, make_dspy_text_extractor_signature,
)
from llm_synthesis.transformers.synthesis_extraction.dspy_synthesis_extraction import (
    DspySynthesisExtractor, make_dspy_synthesis_extractor_signature,
)

configure_dspy("gemini-2.0-flash")
paper_text = open("my_paper.txt").read()

# Step 1: find materials
material_extractor = DspyTextExtractor(
    signature=make_dspy_text_extractor_signature(
        instructions="Extract all synthesized materials as chemical formulas.",
        output_description="Comma-separated list of material formulas.",
    ),
    lm=get_llm_from_name("gemini-2.0-flash", model_kwargs={"temperature": 0.0}),
)
materials = [
    m.strip()
    for m in material_extractor.forward(input=paper_text).split(",")
    if m.strip()
]
print("Materials:", materials)

# Step 2: extract synthesis for the first material
synthesis_extractor = DspySynthesisExtractor(
    signature=make_dspy_synthesis_extractor_signature(
        instructions="Extract the complete synthesis procedure."
    ),
    lm=get_llm_from_name("gemini-2.0-flash", model_kwargs={"temperature": 0.0}),
)
synthesis = synthesis_extractor.forward(input=(paper_text, materials[0]))
print(json.dumps(synthesis.model_dump(), indent=2))
```

---

## Understanding the output

Each result file looks like this (simplified):

```json
{
  "material": "Fe2O3",
  "synthesis": {
    "target_compound": "Fe2O3",
    "synthesis_method": "hydrothermal",
    "starting_materials": [
      {"name": "FeCl3", "amount": 1.62, "unit": "g", "purity": "98%"}
    ],
    "steps": [
      {"step_number": 1, "action": "dissolve", "conditions": {"temperature": 25, "temp_unit": "C"}},
      {"step_number": 2, "action": "heat",    "conditions": {"temperature": 180, "temp_unit": "C", "duration": 12, "time_unit": "h"}}
    ]
  },
  "evaluation": {
    "overall_score": 4.2
  }
}
```

For a full explanation of every field, see [Output Format](../user-guide/output-format.md).

---

## Next steps

| Goal | Resource |
|---|---|
| Process many papers | `lemat-synth batch` or deployment scripts |
| Also extract performance plots | `lemat-synth batch ... --with-performance` |
| Change the LLM | [Configuration guide](../user-guide/configuration.md) |
| Understand all scripts and notebooks | [examples/README.md](../../examples/README.md) |
