# Troubleshooting

This page covers the most common problems encountered when running LeMat-Synth,
with plain-English explanations and fixes.

---

## Installation problems

### `ModuleNotFoundError: No module named 'llm_synthesis'`

**Cause:** The package has not been installed into your virtual environment.

**Fix:**
```bash
uv sync
uv pip install -e .
uv run python -c "import llm_synthesis"   # should produce no output
```

---

### `ModuleNotFoundError: No module named 'uv'` or `command not found: uv`

**Cause:** `uv` is not installed on your system.

**Fix:** Install it following the [official uv instructions](https://github.com/astral-sh/uv?tab=readme-ov-file#installation):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

---

### `playwright install` errors

**Cause:** Playwright (used for downloading PDFs via browser) requires a one-time
browser binary download that must be run separately.

**Fix:**
```bash
uv run playwright install
```

---

## API key problems

### Empty results, blank output, or `AuthenticationError`

**Cause:** The required API key is not set or not being loaded.

**Fix:**
1. Check that your `.env` file exists at the repository root and contains the key:
   ```
   GEMINI_API_KEY=your-key-here
   ```
2. If running a script directly, make sure to `source .env` first (Linux/macOS):
   ```bash
   source .env
   uv run examples/scripts/deployment/extract_synthesis_procedure_from_text.py
   ```
3. The `lemat-synth` CLI and the notebooks load `.env` automatically.

**Which key do I need?**

| Task | Required key |
|------|-------------|
| Synthesis extraction (default) | `GEMINI_API_KEY` |
| Performance plot extraction | `ANTHROPIC_API_KEY` (Claude reads plots) |
| Mistral OCR for PDF extraction | `MISTRAL_API_KEY` |
| OpenAI models | `OPENAI_API_KEY` |
| Qwen via OpenRouter | `OPENROUTER_QWEN_API_KEY` |

---

### `GEMINI_API_KEY not found in .env` (raised by the deployment script)

**Cause:** The script checks explicitly for `GEMINI_API_KEY` and raises if it is missing.

**Fix:** Add it to your `.env` file and `source .env`, or set it as an environment variable:
```bash
export GEMINI_API_KEY=your-key-here
```

---

## PDF extraction problems

### PDF extraction produces no output or very short text

**Possible causes and fixes:**

1. **Docling fails on a corrupted or image-only PDF** — try Mistral OCR instead:
   ```bash
   uv run examples/scripts/deployment/extract_text_from_pdfs.py \
     # edit the script and change PDFExtractorEnum to MISTRAL
   ```
   Or, if you have the Mistral key, convert with Mistral directly.

2. **`playwright install` was not run** — see the installation fix above.
   Playwright is used when downloading PDFs from journal websites.

3. **The PDF is behind a paywall** — the tool cannot access paywalled content
   automatically. Download the PDF manually and pass the local path.

---

## Extraction quality problems

### The extractor returns an empty material list (`[]`)

**Possible causes:**
- The paper text is too short or was extracted incorrectly (very short PDFs, images only)
- The material name is very generic (e.g. "catalyst") and the LLM filters it out
- The paper is in a language the model struggles with

**Fixes:**
- Verify the paper text is complete: open the `.txt`/`.md` file and check it has
  the full synthesis section
- In the quickstart notebook, try printing `len(paper_text)` — it should be > 2,000 characters
- If the text is fine but materials are missed, try a more capable model:
  ```python
  # In the notebook, change:
  lm = configure_dspy("gemini-2.5-flash")  # instead of flash-lite
  ```

---

### Synthesis extraction returns `None` or a synthesis with all `null` fields

**Cause:** The LLM could not parse a valid `GeneralSynthesisOntology` from the text.
This can happen when the paper describes synthesis very briefly, or uses non-standard
terminology.

**Fixes:**
- Check the quality score in `evaluation.overall_score` — if it is below 2.0 the
  extraction likely failed
- Try a more capable model (`gemini-2.5-flash` or `claude-sonnet-4.6`)
- Check that the material name exactly matches how it appears in the paper

---

### Quality scores are low (below 3.0) even for a well-written synthesis

**Cause:** This is normal for brief synthesis descriptions or supplementary-only
procedures. The judge scores relative to what information is present in the source text,
not relative to a hypothetical ideal synthesis.

**It is not a bug.** A score of 3.0 / 5.0 simply means the paper did not provide
enough detail to fill all schema fields — which is faithfully reflected.

---

## Performance / speed problems

### Extraction is very slow or hits rate limits

**Symptoms:** Long waits between papers, `RateLimitError` from the API, or
`429 Too Many Requests`.

**Fixes:**

1. Reduce concurrent LLM calls by setting an environment variable in `.env`:
   ```
   LLM_SYNTHESIS_MAX_CONCURRENT_LLM_CALLS=3
   ```

2. Use a faster / cheaper model for the first pass:
   ```bash
   lemat-synth batch /papers/ results/ --model gemini-2.5-flash-lite
   ```

3. For the deployment scripts, reduce the `ThreadPoolExecutor` `max_workers` argument
   directly in the script.

---

### Figure extraction produces no figures or very few

**Cause:** The figure segmentation models (Florence-2 or DINO) may fail to download
their weights on first run, or GPU memory may be insufficient.

**Fixes:**
- On first run, the model weights are downloaded from HuggingFace. Ensure you have
  internet access and `HF_TOKEN` set if the model is gated.
- If GPU memory is the issue, the extractor falls back to CPU automatically.
  This is slow but correct.
- If segmentation consistently fails, the original full figure (unsegmented) is returned
  as a fallback — extraction will still work, but sub-figure panels will not be split.

---

## Configuration / Hydra problems

### Results are written to an unexpected directory

**Cause:** Hydra automatically creates a timestamped output directory under `results/`
(e.g. `results/single_run/2025-05-13/14-32-01/`).

**To control the output directory**, override `hydra.run.dir` on the command line:
```bash
uv run examples/scripts/deployment/extract_synthesis_procedure_from_text.py \
  hydra.run.dir=my_results/run1
```

---

### `HydraException: Could not load config` or `config not found`

**Cause:** The Hydra scripts must be run from the repository root, **not** from within
the `examples/` folder, because Hydra resolves config paths relative to the working
directory.

**Fix:** Always run from the repository root:
```bash
cd /path/to/lematerial-llm-synthesis
uv run examples/scripts/deployment/extract_synthesis_procedure_from_text.py
```

---

### `omegaconf.errors.ConfigAttributeError: Key ... not in struct`

**Cause:** You tried to set a config key that does not exist in the YAML schema.

**Fix:** Check the exact key name in the relevant YAML file under `examples/config/`.
Use `.` notation to navigate nested keys, e.g.:
```bash
uv run ... data_loader.architecture.data_dir=/my/path  # correct
uv run ... data_loader.data_dir=/my/path               # wrong: key is inside 'architecture'
```

---

## Getting more help

- Check the [examples/README.md](../../examples/README.md) to confirm you are using
  the right script for your use case.
- If the paper text is good but extraction fails consistently, open an issue on GitHub
  with the paper ID, the error message, and the first 200 characters of the paper text.
