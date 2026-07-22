# Superconductor Case Study — Qwen VLM Pipeline Plan

## Goal
Reproduce panels b, c, d of the draft figure using the HF dataset
(`LeMaterial/LeMat-Synth-Papers`, split `superconductor_keywords_and_LLM`)
instead of re-extracting text from scratch, and swap the VLM from Claude
Sonnet 4.6 to Qwen (`qwen3.5-397b-a17b`, via OpenRouter).

## What changed from the original pipeline

1. **Materials**: read directly from the `structured_synthesis` column
   (already in the HF dataset) instead of a fresh Gemini material-extraction
   call. No LLM cost for this step.
2. **Synthesis procedures**: already in `structured_synthesis` /
   `text_paper` / `text_si` columns — not re-extracted.
3. **Tc reading + material linking**: merged into a single Qwen VLM call
   per figure (`TcVLMProcessor.process_from_figures`). The prompt gets the
   paper's material list and is asked to both (a) read Tc via geometric
   construction and (b) match each curve to a material name, in one pass.
   This replaces the old two-step "digitize plot -> DeepSeek link series to
   material" pipeline entirely.
4. **Figure extraction**: OCR (Mistral) + figure segmentation (Florence-2,
   LoRA-tuned) — the only step that still needs to run fresh per paper,
   since the HF dataset's `images` column is empty for arxiv-sourced rows
   (see prior investigation: images were only ever populated for
   chemrxiv-sourced papers, never arxiv).
5. **Figure cache**: segmented figures (base64 + metadata) are cached to
   `<output_dir>/_figure_cache/<paper_id>/figures.json` after first
   extraction. Re-running with a different VLM/model/prompt reuses the
   cache and skips OCR + Florence entirely.

## Bugs fixed along the way

- `LLM_REGISTRY` (`src/llm_synthesis/utils/llms.py`): Qwen/Kimi OpenRouter
  configs used `extra_kwargs={"enable_thinking": False}`, which OpenRouter
  silently ignores for these models — they'd burn the entire `max_tokens`
  budget on hidden reasoning tokens and return empty content. Fixed to
  `extra_kwargs={"reasoning": {"enabled": False}}`, the correct OpenRouter
  param. Verified fix with a direct litellm call (`reasoning_tokens` went
  from 95/100 tokens burned to 0).
- Florence-2 runs on CPU, not GPU, due to an NVIDIA driver/CUDA version
  mismatch (driver reports CUDA 12020, torch build installed expects newer).
  `nvidia-smi` shows 8 working GPUs, but `torch.cuda.is_available()` is
  `False`. Not fixed yet — deferred per your call to prioritize VLM
  parallelization first (see Open Items).

## Scripts

- `download_pdfs.py` — fetches arxiv PDFs for the HF dataset split.
  `--max N` for the first N rows, `--ids <id> <id> ...` for specific dataset
  ids (handles old-style ids with a `/`, e.g. `cond-mat/0102313`, by
  sanitizing to `cond-mat_0102313.pdf` on disk).
- `run_from_hf.py` — the main pipeline. Per paper: load materials from
  `structured_synthesis` -> OCR + Florence figure extraction (cached) ->
  Qwen VLM Tc-reading+linking -> write `tc_master.csv` (same schema as the
  original Claude-based pipeline, so results are directly comparable).
  `--workers N` controls paper-level concurrency (Mistral OCR + Qwen calls
  parallelize across papers; Florence-2 is serialized via a lock since the
  model instance isn't thread-safe for concurrent CPU inference).

## Runs in progress (launched concurrently, since they touch disjoint
## PDF/output directories and the machine has 52 cores / 502GB RAM to
## spare)

1. **verify5** — 5 papers (already-downloaded pilot set), `--workers 4`.
   Purpose: confirm the parallelized + cached pipeline still produces
   correct output after the material-matching rework and Qwen registry fix.
2. **snippet19** — the 19 papers from the original Claude-based
   `tc_master_snippet.csv` (`results/results_superconductors/`), which
   already has human/text-extracted `tc_text` values alongside Claude's
   `tc_vlm_orig`. Running these through the Qwen pipeline gives an
   apples-to-apples Qwen-vs-human comparison for panel b, without needing
   any new ground-truth annotation (the "human" values already exist in
   this file — I initially thought I needed a separate annotation file on
   your local machine, but this snippet CSV already has what's needed).
3. **trial100** — 100 papers (already downloaded), `--workers 6`. Broader,
   descriptive-only (no ground truth) coverage for panels c/d.

## Time estimates

Based on the single-paper pilot actually measured before parallelization:
- Florence-2 (CPU): ~9 min / paper for ~20-25 figures (fixed cost per
  paper, not reduced by parallelization — only by fixing the GPU driver).
- Qwen VLM, sequential (pre-fix): ~3.7 min/figure x 23 figures = ~85 min.
- Qwen VLM, parallelized (8 concurrent workers within a paper): expected
  ~85/8 ≈ ~11 min/paper.
- **Per-paper total (post-fix, single paper): ~20 min** (9 min Florence +
  11 min Qwen), down from ~94 min measured pre-fix.

For a batch of N papers with `--workers W` (paper-level parallelism):
- Florence-2 is serialized (one paper's segmentation at a time) ->
  Florence-2 wall-clock ≈ N x 9 min regardless of W.
- Qwen VLM calls overlap across papers (each paper's own 8-worker figure
  pool, papers themselves also run W-wide) -> Qwen wall-clock roughly
  (N x 11 min) / W, though real speedup depends on OpenRouter rate limits.
- **100 papers, `--workers 6`: rough estimate ~5-7 hours total**, dominated
  by the serialized Florence-2 step (~15 hours of Florence time alone if
  every paper has 23 figures — likely an overestimate since 23 was a
  figure-heavy outlier; a more typical paper probably has 5-10 quantitative
  figures, i.e. ~2-4 min Florence + ~2-5 min Qwen per paper, plausibly
  ~3-4 hours for 100 papers at `--workers 6`).
- These are estimates, not measurements. The three runs launched above will
  give real numbers to replace this section once they complete.

## Cost estimates

No cost was tracked in the original pilot run (bug: `TcVLMProcessor` didn't
track litellm cost). Fixed — `TcVLMProcessor.get_cost()` now sums
`litellm.completion_cost()` across all VLM calls, thread-safe via a lock,
and `run_from_hf.py` logs it per-paper and totals it in the batch summary.

**Mistral OCR cost is NOT currently tracked in code** — `MistralPDFExtractor`
has no cost accounting at all (only `TcVLMProcessor.get_cost()` tracks the
Qwen VLM side). Mistral's published OCR pricing is ~$1 per 1000 pages
($0.001/page); a typical paper (~10-20 pages) is roughly **$0.01-0.02/paper**,
one-time since figures are cached after first extraction. Small relative to
VLM cost, but omitted from the `get_cost()` total below — worth adding
tracking if precise cost accounting matters later.

Rough estimate before real numbers land (Qwen3.5-397B-A17B OpenRouter
pricing is MoE-cheap, order of magnitude ~$0.15/1M input, ~$0.85/1M
output tokens; a single Tc-reading call with materials list + one figure
image is roughly 2-5K tokens in, a few hundred out):
- **~$0.01-0.03 per figure call** (Qwen VLM only)
- **~$0.10-0.50 per paper** (5-15 quantitative figures/paper, typical) +
  ~$0.01-0.02/paper Mistral OCR (untracked, one-time)
- **100 papers: ~$11-52 total** (Qwen + estimated OCR) — should be
  confirmed against real `get_cost()` numbers from the trial100 run once
  it completes; OCR portion will still need to be added by hand since it
  isn't code-tracked.

## Panel-by-panel plan

### Panel b — Tc VLM vs human text Tc
- Ground truth already exists: `results/results_superconductors/
  tc_master_snippet.csv`, columns `tc_text` (human/text-extracted) and
  `tc_vlm_orig` (Claude Sonnet 4.6 VLM), for 19 papers / 93 material rows.
- Once `snippet19` run completes: join its `tc_vlm` column (Qwen) against
  the same file's `tc_text` column on `(paper_id, material)`, recompute
  R²/MAE/RMSE/MAPE and the TP/FN/"not written"/"not SC" counts the same way
  the existing panel-b cell does, just swapping the VLM source.
- Output: same scatter plot style, now Qwen-vs-human instead of
  Claude-vs-human. Directly resolves the panel's "needs consistent format
  with other figures" WIP note (both b panels — VLM-orig and VLM-snippet —
  can now be regenerated from one consistent pipeline).

### Panel c — Materials x synthesis method -> Tc
- `synthesis_method` needs to be threaded through from `structured_synthesis`'s
  `recipe.synthesis_method` field into the CSV writer's output (currently
  dropped since `run_from_hf.py` passes `entry.synthesis=None` to avoid
  needing a `GeneralSynthesisOntology` object). Small addition: extract
  `synthesis_method` as a plain string when building `materials_by_id` and
  pass it through to the CSV row, bypassing the full ontology object.
- Once wired: reuse the existing `visualisation_tc.ipynb` boxplot cell
  (`synthesis_method_tc_boxplot`) against `trial100`'s `tc_master.csv`.

### Panel d — Tc vs year (scale-up demonstration)
- No new wiring needed — `year` is already derived from the arxiv ID in
  `CsvMasterWriter`. Just re-run the existing plotting cell against
  `trial100`'s output. This is the panel that directly demonstrates
  "scale up to all of arXiv" — 100 papers is the proof-of-concept slice
  before scaling to the full 1384-paper split (or beyond, per the
  dataset's arxiv coverage).

## Open items (not yet done)

1. **Florence-2 GPU fix** — deferred. Needs either an NVIDIA driver update
   or a torch/CUDA build matching the installed driver (reports CUDA
   12020). This is the single biggest remaining speed lever; without it,
   Florence-2's ~9 min/paper CPU cost is now the dominant serialized
   bottleneck (Qwen calls parallelize away, Florence doesn't).
2. **`synthesis_method` wiring** for panel c — not yet implemented (see
   above), small change once the 100-paper trial's core pipeline is
   confirmed working.
3. **Scaling beyond 100 papers** — once panels b/c/d are validated on this
   trial, decide whether to run the remaining ~1284 papers in the HF
   dataset's superconductor split, and re-estimate time/cost from the
   trial100 run's real numbers rather than the rough estimates above.
