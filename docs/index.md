# LeMat-Synth

**An open-source multi-modal toolbox for extracting structured synthesis procedures
and performance data from materials science literature at scale.**

LeMat-Synth is the implementation of [LeMat-Synth v1.0](https://arxiv.org/abs/2510.26824)
(NeurIPS AI4Mat 2025). Given a materials science paper, it automatically extracts:

- **Synthesized materials** — identified by chemical formula
- **Synthesis procedures** — step-by-step, structured by a controlled ontology
- **Performance data** — quantitative values read from plots, linked to materials

[![Paper](https://img.shields.io/badge/arXiv-2510.26824-b31b1b.svg)](https://arxiv.org/abs/2510.26824)
[![Dataset](https://img.shields.io/badge/🤗%20HuggingFace-Dataset-yellow)](https://huggingface.co/datasets/LeMaterial/LeMat-Synth)

---

## How it works

```
Your paper (PDF or text)
        │
        ▼
 ┌─────────────────┐
 │ Material        │  "Which materials were synthesized?"
 │ Extraction      │  → ["Fe2O3", "Ni/Fe2O3"]
 └────────┬────────┘
          │ for each material
          ▼
 ┌─────────────────┐
 │ Synthesis       │  "How was Fe2O3 made?"
 │ Extraction      │  → steps, conditions, reagents, equipment
 └────────┬────────┘
          │
          ▼
 ┌─────────────────┐
 │ Judge           │  Quality score (1–5) per dimension
 └────────┬────────┘
          │ (optional: --with-performance)
          ▼
 ┌─────────────────┐
 │ Figure          │  Finds plots in the paper
 │ Extraction      │
 └────────┬────────┘
          │
          ▼
 ┌─────────────────┐
 │ Plot Data       │  Reads x/y coordinates from each plot
 │ Extraction      │
 └────────┬────────┘
          │
          ▼
 ┌─────────────────┐
 │ Performance     │  Links plot series to synthesized materials
 │ Linking         │
 └─────────────────┘
          │
          ▼
   result JSON files
```

---

## Quick start

```bash
# Install
git clone https://github.com/LeMaterial/lematerial-llm-synthesis.git
cd lematerial-llm-synthesis
uv sync && uv pip install -e .

# Add your API key to .env
echo "GEMINI_API_KEY=your_key_here" >> .env

# Extract synthesis from one paper
lemat-synth extract my_paper.txt

# Or process a whole folder of PDFs
lemat-synth batch /path/to/pdfs/ results/ --domain catalysis
```

---

## Where to go next

- **New to the tool?** Start with the [Quickstart guide](getting-started/quickstart.md)
  or open `examples/notebooks/00_quickstart.ipynb` in Jupyter.
- **Have PDFs?** See the [examples/README.md](https://github.com/LeMaterial/lematerial-llm-synthesis/blob/main/examples/README.md) decision table to
  pick the right script.
- **Don't understand the output?** Read [Output Format](user-guide/output-format.md).
- **Want to change the LLM or use local files?** See [Configuration](user-guide/configuration.md).
- **Something broken?** See [Troubleshooting](user-guide/troubleshooting.md).
- **Want to use the Python API?** See the [API Reference](api/pipeline.md).
- **Want to contribute or extend the pipeline?** Start with the
  [Developer Guide](developer-guide/architecture.md).

---

## Citation

If you use LeMat-Synth in your research, please cite:

```bibtex
@article{lederbauer2025lemat,
  title={LeMat-Synth: a multi-modal toolbox to curate broad synthesis procedure
         databases from scientific literature},
  author={Lederbauer, Magdalena and others},
  journal={arXiv preprint arXiv:2510.26824},
  year={2025}
}
```
