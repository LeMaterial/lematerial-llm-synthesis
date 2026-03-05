# Prefect Integration Design

## Architecture Overview

```
Entry Point: extract_synthesis_prefect.py
     │
     │ @hydra.main() — loads config, resolves all paths to absolute
     │ OmegaConf.to_container(cfg, resolve=True) → plain dict
     ▼
synthesis_extraction_flow(config: dict)              [@flow]
     │ ThreadPoolTaskRunner(max_workers=N)
     │ Instantiates components from config dict via hydra.utils.instantiate()
     │
     ├─── load_papers(data_loader) ─────────────────── [@task, no retries]
     │         │ Returns: list[Paper]
     │         ▼
     ├─── For each Paper (concurrent via ThreadPoolTaskRunner):
     │         │
     │         ├── extract_materials(extractor, paper_text)  [@task, retries=3]
     │         │       │ Returns: list[str]
     │         │       ▼
     │         ├── For each material:
     │         │       │
     │         │       ├── extract_synthesis(extractor, paper_text, material)
     │         │       │         [@task, retries=3]
     │         │       │         Returns: GeneralSynthesisOntology
     │         │       │
     │         │       └── evaluate_synthesis(judge, paper_text, synthesis_json, material)
     │         │                 [@task, retries=2]
     │         │                 Returns: evaluation dict
     │         │
     │         └── save_paper_results(result_gather, paper_with_results)
     │                   [@task, retries=1]
     │                   Returns: None
     │
     └─── Log total cost
```

## Component Mapping

| Pipeline Step | Current Implementation | Prefect Wrapper | Retry |
|---------------|----------------------|-----------------|-------|
| Load papers | `PaperLoaderInterface.load()` | `load_papers` `@task` | None (deterministic I/O) |
| Extract materials | `MaterialExtractorInterface.forward(input=text)` | `extract_materials` `@task` | 3 retries, 5s delay (LLM call) |
| Extract synthesis | `SynthesisExtractorInterface.forward(input=(text, material))` | `extract_synthesis` `@task` | 3 retries, 5s delay (LLM call) |
| Evaluate synthesis | `DspyGeneralSynthesisJudge.forward((text, json, material))` | `evaluate_synthesis` `@task` | 2 retries, 5s delay (LLM call) |
| Save results | `ResultGatherInterface.gather(paper_with_results)` | `save_paper_results` `@task` | 1 retry (I/O) |
| Orchestration | `ThreadPoolExecutor(max_workers=4)` | `synthesis_extraction_flow` `@flow` | N/A |

## Data Types Between Tasks

| Task | Input Types | Output Type | Notes |
|------|------------|-------------|-------|
| `load_papers` | `PaperLoaderInterface` (from config) | `list[Paper]` | Paper: `src/llm_synthesis/models/paper.py` |
| `extract_materials` | `MaterialExtractorInterface`, `str` (paper_text) | `list[str]` | CSV string parsed to list |
| `extract_synthesis` | `SynthesisExtractorInterface`, `str`, `str` (material) | `GeneralSynthesisOntology` | Pydantic model |
| `evaluate_synthesis` | judge, `str`, `str` (synthesis JSON), `str` (material) | evaluation dict | |
| `save_paper_results` | `ResultGatherInterface`, `PaperWithSynthesisOntologies` | `None` | |

### Critical API Notes

**`extract_materials`**: `forward()` takes keyword arg `input`, returns comma-separated string.
```python
result = extractor.forward(input=clean_text(paper_text))
materials = [m.strip() for m in result.replace('\n', ',').split(',') if m.strip()]
# Reference: extract_synthesis_procedure_from_text.py:142-155
```

**`extract_synthesis`**: `forward()` takes tuple `(text, material)` as `input`.
```python
result = extractor.forward(input=(clean_text(paper_text), material))
```

**`evaluate_synthesis`**: `forward()` takes positional tuple `(text, synthesis_json, material)`.
```python
result = judge.forward((clean_text(paper_text), synthesis_json, material))
```

## Retry Strategy

| Task | Retries | Delay | Rationale |
|------|---------|-------|-----------|
| `load_papers` | 0 | — | Deterministic; HuggingFace cached locally after first load |
| `extract_materials` | 3 | 5s | LLM API call; transient rate-limit errors expected |
| `extract_synthesis` | 3 | 5s | LLM API call; most expensive step |
| `evaluate_synthesis` | 2 | 5s | LLM API call; less critical than extraction |
| `save_paper_results` | 1 | 5s | I/O; once is usually enough, retry covers transient FS errors |

Retry config is driven from Hydra `orchestration/default.yaml` and passed to flow:
```python
@task(retries=cfg["orchestration"]["retries"]["material_extraction"],
      retry_delay_seconds=cfg["orchestration"]["retry_delay_seconds"])
```

## Thread Safety Analysis

### Problem: `SystemPrefixedLM._cumulative_cost_usd`

**Location:** `src/llm_synthesis/utils/llms.py:139`

`SystemPrefixedLM` maintains a shared `_cumulative_cost_usd: float` that is incremented in
`_extract_and_accumulate_cost()` without any locking. Under `ThreadPoolTaskRunner`, multiple
Prefect tasks run concurrently in threads — if they share the same `SystemPrefixedLM` instance
(which happens when components are instantiated once and reused), the cost accumulation is a
**race condition**:

```python
# Current (NOT thread-safe):
self._cumulative_cost_usd += delta  # Read-modify-write without lock
```

**Fix:** Add `threading.Lock` to `SystemPrefixedLM.__init__` and wrap the increment:

```python
import threading

class SystemPrefixedLM:
    def __init__(self, ...):
        ...
        self._cost_lock = threading.Lock()
        self._cumulative_cost_usd = 0.0

    def _extract_and_accumulate_cost(self, ...):
        ...
        with self._cost_lock:
            self._cumulative_cost_usd += delta
```

**DSPy settings thread safety:** `dspy.settings.context()` is already used per-call and is
thread-safe. No fix needed there.

## Serialization Strategy

### Problem: Non-picklable Objects

Prefect serializes task arguments when using `ThreadPoolTaskRunner`. Objects like `dspy.LM`
contain `httpx.Client` connection pools, which are NOT picklable. Pre-instantiated
`SynthesisExtractorInterface`, `MaterialExtractorInterface`, and `DspyGeneralSynthesisJudge`
objects would fail serialization.

### Solution: Instantiate Inside Flow from Config Dict

1. **Entry point** converts Hydra `DictConfig` → plain Python `dict`:
   ```python
   from omegaconf import OmegaConf
   config_dict = OmegaConf.to_container(cfg, resolve=True)
   ```

2. **Flow** receives plain `dict` (fully serializable) and instantiates components
   via `hydra.utils.instantiate()`:
   ```python
   @flow
   def synthesis_extraction_flow(config: dict):
       data_loader = hydra.utils.instantiate(config["data_loader"])
       material_extractor = hydra.utils.instantiate(config["material_extraction"])
       synthesis_extractor = hydra.utils.instantiate(config["synthesis_extraction"])
       judge = hydra.utils.instantiate(config["judge"])
       result_gather = hydra.utils.instantiate(config["result_save"])
       ...
   ```

3. **Tasks** receive already-instantiated component objects as arguments (passed within the
   same process — no cross-process serialization for `ThreadPoolTaskRunner`).

**Note:** `ThreadPoolTaskRunner` runs in threads (not subprocesses), so the actual pickling
constraint is relaxed — objects don't need to be picklable for thread-based task runners.
However, keeping instantiation inside the flow is still best practice for clarity and future
compatibility with `ProcessPoolTaskRunner`.

## Path Resolution

Hydra changes CWD to the output directory (`hydra.job.chdir: true`). All relative paths
must be resolved to absolute BEFORE calling the flow.

| Path | Config location | Resolution |
|------|----------------|------------|
| `system_prompt` | `synthesis_extraction.system_prompt` | `os.path.join(get_original_cwd(), path)` |
| `data_dir` | `data_loader.data_dir` (if local) | `os.path.join(get_original_cwd(), path)` |
| `result_dir` | `result_save.result_dir` | `os.path.join(get_original_cwd(), path)` |

Reference: `extract_synthesis_procedure_from_text.py:64-85` (system_prompt resolution),
`extract_synthesis_procedure_from_text.py:47-57` (data_dir), lines 142-155 (material parsing).

## Key Integration Points (file:line references)

| Component | File | Location |
|-----------|------|----------|
| `ExtractorInterface` base class | `src/llm_synthesis/transformers/base.py` | L16 |
| `BasePipeline` ABC | `src/llm_synthesis/services/pipelines/base_pipeline.py` | L4 |
| `SynthesisPerformancePipeline` (6-step reference) | `src/llm_synthesis/services/pipelines/synthesis_performance_pipeline.py` | L1 |
| `SystemPrefixedLM._extract_and_accumulate_cost` | `src/llm_synthesis/utils/llms.py` | L139 |
| `LLMConfig`, `LLM_REGISTRY` | `src/llm_synthesis/utils/llms.py` | L44 |
| `configure_dspy`, `get_llm_from_name` | `src/llm_synthesis/utils/dspy_utils.py` | L1 |
| `Paper`, `SynthesisEntry`, `PaperWithSynthesisOntologies` | `src/llm_synthesis/models/paper.py` | L1 |
| Material extraction call pattern | `examples/scripts/deployment/extract_synthesis_procedure_from_text.py` | L142-155 |
| System prompt path resolution | `examples/scripts/deployment/extract_synthesis_procedure_from_text.py` | L64-85 |
| Hydra root config | `examples/config/config.yaml` | L1 |
| `FSResultGather` | `src/llm_synthesis/result_gather/synthesis_results/fs_result_gather.py` | L1 |
