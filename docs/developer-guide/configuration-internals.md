# Configuration Internals

This page explains how the Hydra configuration system works under the hood.
It is aimed at developers who want to add new config variants, change how
components are wired together, or debug configuration issues.

Users who only want to swap an LLM or point to a different dataset can stop at
the [user-facing configuration guide](../user-guide/configuration.md).

---

## How Hydra instantiates components

Every pipeline component is described in YAML using a `_target_` key:

```yaml
architecture:
  _target_: llm_synthesis.transformers.synthesis_extraction.dspy_synthesis_extraction.DspySynthesisExtractor
  lm:
    _target_: llm_synthesis.utils.dspy_utils.get_llm_from_name
    llm_name: "gemini-2.0-flash"
    model_kwargs:
      temperature: 0.0
      max_tokens: 12000
  signature:
    _target_: llm_synthesis.transformers.synthesis_extraction.dspy_synthesis_extraction.make_dspy_synthesis_extractor_signature
    instructions: "Extract the structured synthesis for a specific material."
```

At runtime, `hydra.utils.instantiate(cfg.architecture)` does the following:

1. Resolves `_target_` to a Python callable (class or function).
2. Recursively instantiates any nested dicts that also have `_target_`.
3. Passes all remaining keys as keyword arguments to the callable.

This means **the YAML file is the dependency injection container**. Swapping
`_target_` replaces the implementation; changing other keys changes the
constructor arguments — all without touching Python.

---

## The config composition model

`examples/config/config.yaml` is the root. Its `defaults:` list names one
YAML file per config group:

```yaml
defaults:
  - _self_
  - data_loader: default
  - synthesis_extraction: default
  - material_extraction: default
  - judge: default
  - result_save: default
  - plot_extraction: default
```

Each entry `group: variant` tells Hydra to load
`examples/config/<group>/<variant>.yaml` and merge it into the global config.
`_self_` means "the values defined directly in this file take precedence over
the group defaults".

**Hydra merges configs** — keys in a group file are placed under the group
name in the global config. So `synthesis_extraction/default.yaml` becomes
accessible at `cfg.synthesis_extraction` in Python.

---

## Config group reference

### `data_loader/`

Controls which papers are loaded and how many.

| File | What it does |
|------|-------------|
| `default.yaml` | Streams from HuggingFace (`LeMat-Synth-Papers`, split configurable) |
| `local.yaml` | Reads `.txt` files from a local directory |
| `annotation.yaml` | HF stream restricted to papers present in `annotations/` |

All loaders expose `number_of_samples` at the top level of the group config.
Set it to `null` to process everything, or to an integer to cap the run.

**Relevant `_target_` classes:**

| Class | File |
|-------|------|
| `HFLoader` | `src/llm_synthesis/data_loader/paper_loader/hf_paper_loader.py` |
| `FsPaperLoader` | `src/llm_synthesis/data_loader/paper_loader/fs_paper_loader.py` |
| `AnnotationHFLoader` | `src/llm_synthesis/data_loader/paper_loader/annotation_hf_paper_loader.py` |

---

### `synthesis_extraction/` and `material_extraction/`

Both share the same structure. The `default.yaml` variant uses a single LLM;
`multi_llm.yaml` runs an ensemble and stores per-LLM outputs.

Key fields:

```yaml
architecture:
  _target_: ...DspySynthesisExtractor
  signature:
    _target_: ...make_dspy_synthesis_extractor_signature
    instructions: "..."          # the task description in the prompt
    output_description: "..."    # description of the expected output field
  lm:
    _target_: ...get_llm_from_name
    llm_name: "gemini-2.0-flash" # ← change this to swap the LLM
    model_kwargs:
      temperature: 0.0
      max_tokens: 12000
      num_retries: 3
    system_prompt:
      _target_: llm_synthesis.utils.read_prompt_str_from_txt
      prompt_path: "examples/system_prompts/synthesis_extraction/default.txt"
```

The `system_prompt` is loaded from a plain `.txt` file at runtime. Editing
that file changes the LLM's persona and task framing without touching YAML or
Python.

**Multi-LLM variant** adds `llm_names: [...]` — the pipeline runs the
extractor once per model and stores all outputs keyed by model name.

---

### `judge/`

The judge evaluates extraction quality by prompting an LLM to score the output
against seven criteria. Key fields beyond the standard `lm` block:

```yaml
enable_reasoning_traces: true   # store the judge's chain-of-thought
confidence_threshold: 0.7       # minimum score to accept an extraction
```

The `linking.yaml` variant configures a separate judge specifically for the
plot-to-material linking task.

---

### `result_save/`

Controls where and in what format results are written. Currently delegates to
`FsResultGather` (local filesystem). The `multi_llm.yaml` variant writes one
result file per LLM in addition to the merged output.

---

### `plot_extraction/`

Configures the visual LLM stack for reading data off charts. Notably exposes
retry counts, the ranking metric used to select the best plot read among
multiple VLM attempts, and which VLMs to use.

---

## Runtime overrides

Hydra supports dot-separated key overrides from the command line. The syntax
is `group.subkey=value`:

```bash
# Override a nested value inside a group
uv run python ... synthesis_extraction.architecture.lm.llm_name="claude-opus-4-7"

# Switch an entire group to a named variant
uv run python ... synthesis_extraction=multi_llm

# Override a top-level key
uv run python ... number_of_samples=10

# Change the output directory
uv run python ... hydra.run.dir=my_results/debug
```

Overrides are applied after all YAML files are merged, so they always win.

---

## Adding a new config variant

1. Copy an existing YAML file in the relevant group directory:
   ```bash
   cp examples/config/synthesis_extraction/default.yaml \
      examples/config/synthesis_extraction/my_variant.yaml
   ```
2. Edit `_target_` to point at your new class (if you wrote one), or change
   `llm_name` / other fields as needed.
3. Select it at runtime:
   ```bash
   uv run python ... synthesis_extraction=my_variant
   ```

No changes to `config.yaml` or Python are needed.

---

## Adding a new config group

If you add an entirely new pipeline stage, create a new directory under
`examples/config/` and add a `defaults:` entry in `config.yaml`:

```yaml
defaults:
  - ...existing entries...
  - my_new_stage: default       # loads examples/config/my_new_stage/default.yaml
```

Then create `examples/config/my_new_stage/default.yaml` with `_target_`
pointing at your stage's class.

---

## Debugging config composition

To print the fully-merged config without running anything, append
`--cfg job` to any script call:

```bash
uv run python examples/scripts/deployment/extract_synthesis_procedure_from_text.py \
    synthesis_extraction=multi_llm --cfg job
```

This outputs the exact config object that the script will see, after all
merges and overrides. Use it to verify that your overrides are applied
correctly before a long run.
