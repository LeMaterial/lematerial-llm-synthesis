"""Extract synthesis recipes and judge them over a LeMat-Synth-Papers split.

Fixed to the models used in the agreement analysis:
  - extractor: claude-sonnet-4.6   (Anthropic key, ANTHROPIC_API_KEY)
  - judge:     deepseek-v3.2       (OpenRouter, both keys round-robined)

Per paper it builds the context from title + abstract + text_paper (+ text_si
only when non-empty, truncated), runs material extraction then per-material
synthesis extraction with Claude, and scores each extraction with deepseek on
the 8 rubric dimensions. Results stream to a JSONL file (crash-safe, resumable
by id) and can optionally be pushed to the Hub as a NEW dataset repo.

Concurrency: a pool of N isolated pipeline instances (each with its own LM
objects, so no shared dspy history/state races); the two OpenRouter keys are
alternated across the pool. Pool size == max in-flight papers.

This reuses the exact Hydra components from examples/config (material_extraction
/ synthesis_extraction / judge defaults) with only the model names overridden,
so extraction/judge behaviour matches the analysis.

Usage (run from the repo root; needs `datasets`):
    uv run --with datasets python \
        examples/scripts/deployment/run_extract_judge_hf.py \
        --config superconductor_keywords_and_LLM --split full \
        --limit 10 --concurrency 8            # smoke test, local JSONL only

    # full split, then push results to a NEW hub repo (does not touch source):
    uv run --with datasets python \
        examples/scripts/deployment/run_extract_judge_hf.py \
        --config superconductor_keywords_and_LLM --split full \
        --concurrency 32 --push --hub-repo <user>/LeMat-recipes
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
os.chdir(REPO_ROOT)  # config prompt_path values are relative to the repo root
sys.path.insert(0, str(REPO_ROOT / "src"))

from hydra.utils import instantiate  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

EXTRACTOR_MODEL = "claude-sonnet-4.6"
JUDGE_MODEL = "deepseek-v3.2"
DATASET_URI = "LeMaterial/LeMat-Synth-Papers"
OPENROUTER_BASE = "https://openrouter.ai/api/v1"

SCORE_COLUMNS = [
    "structural_completeness_score",
    "material_extraction_score",
    "process_steps_score",
    "equipment_extraction_score",
    "conditions_extraction_score",
    "semantic_accuracy_score",
    "format_compliance_score",
    "overall_score",
]

CFG = {
    "material": "examples/config/material_extraction/default.yaml",
    "synthesis": "examples/config/synthesis_extraction/default.yaml",
    "judge": "examples/config/judge/default.yaml",
}


def _load_dotenv():
    """Best-effort .env load so ANTHROPIC/OPENROUTER keys are available."""
    env = REPO_ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        v = v.strip()
        if v and v[0] in "\"'":  # quoted value: take content inside quotes
            v = v[1:].split(v[0], 1)[0]
        else:  # unquoted: drop any trailing inline comment
            v = v.split("#", 1)[0].strip()
        os.environ.setdefault(k.strip(), v)


def _instantiate(path, llm_name):
    cfg = OmegaConf.load(path)
    cfg.architecture.lm.llm_name = llm_name
    return instantiate(cfg.architecture)


def make_pipeline(index, openrouter_keys):
    """Build one isolated (material, synthesis, judge) pipeline.

    The judge's OpenRouter key is chosen round-robin from ``openrouter_keys``.
    """
    material = _instantiate(CFG["material"], EXTRACTOR_MODEL)
    synthesis = _instantiate(CFG["synthesis"], EXTRACTOR_MODEL)
    judge = _instantiate(CFG["judge"], JUDGE_MODEL)
    key = openrouter_keys[index % len(openrouter_keys)]
    # swap the deepseek key so we spread load across both OpenRouter keys
    judge.lm.kwargs["api_key"] = key
    judge.lm.kwargs["api_base"] = OPENROUTER_BASE
    return {
        "material": material,
        "synthesis": synthesis,
        "judge": judge,
        "key_index": index % len(openrouter_keys),
    }


def build_context(row, si_char_cap):
    parts = []
    if (row.get("title") or "").strip():
        parts.append(f"# {row['title'].strip()}")
    if (row.get("abstract") or "").strip():
        parts.append(f"## Abstract\n{row['abstract'].strip()}")
    if (row.get("text_paper") or "").strip():
        parts.append(f"## Paper\n{row['text_paper'].strip()}")
    si = (row.get("text_si") or "").strip()
    if si:
        if si_char_cap and len(si) > si_char_cap:
            si = si[:si_char_cap] + "\n[... SI truncated ...]"
        parts.append(f"## Supporting Information\n{si}")
    return "\n\n".join(parts)


def parse_materials(raw, max_materials):
    seen, out = set(), []
    for m in (raw or "").split(","):
        m = m.strip()
        if m and m.lower() not in seen:
            seen.add(m.lower())
            out.append(m)
        if len(out) >= max_materials:
            break
    return out


def process_paper(row, pipe, si_char_cap, max_materials):
    """Synchronous end-to-end for one paper. Returns a result dict."""
    pid = row.get("id")
    result = {
        "id": pid,
        "title": row.get("title"),
        "source": row.get("source"),
        "pdf_extractor": row.get("pdf_extractor"),
        "extractor_model": EXTRACTOR_MODEL,
        "judge_model": JUDGE_MODEL,
        "judge_key_index": pipe["key_index"],
        "structured_synthesis": None,
        "evaluations": None,
        "n_materials": 0,
        "error": None,
    }
    for c in SCORE_COLUMNS:
        result[c] = None
    try:
        ctx = build_context(row, si_char_cap)
        if not ctx.strip():
            result["error"] = "empty context"
            return result
        materials = parse_materials(
            pipe["material"].forward(ctx), max_materials
        )
        if not materials:
            result["error"] = "no materials extracted"
            result["structured_synthesis"] = json.dumps([])
            return result

        recipes, evals, score_acc = [], [], {c: [] for c in SCORE_COLUMNS}
        for m in materials:
            onto = pipe["synthesis"].forward((ctx, m))
            onto_json = onto.model_dump_json()
            ev = pipe["judge"].forward((ctx, onto_json, m))
            sc = ev.scores
            scores = {c: getattr(sc, c, None) for c in SCORE_COLUMNS}
            recipes.append(
                {"material_name": m, "recipe": json.loads(onto_json)}
            )
            evals.append(
                {
                    "material_name": m,
                    "scores": scores,
                    "reasoning": getattr(ev, "reasoning", None),
                    "confidence_level": getattr(ev, "confidence_level", None),
                }
            )
            for c in SCORE_COLUMNS:
                if isinstance(scores[c], (int, float)):
                    score_acc[c].append(scores[c])

        result["structured_synthesis"] = json.dumps(recipes, ensure_ascii=False)
        result["evaluations"] = json.dumps(evals, ensure_ascii=False)
        result["n_materials"] = len(materials)
        for c in SCORE_COLUMNS:
            vals = score_acc[c]
            result[c] = round(sum(vals) / len(vals), 4) if vals else None
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"[:500]
    return result


def load_done_ids(out_path):
    done = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            try:
                r = json.loads(line)
                # only treat successful rows as done; retry prior errors
                if r.get("id") is not None and r.get("error") is None:
                    done.add(r["id"])
            except json.JSONDecodeError:
                continue
    return done


def iter_rows(config, split, done_ids, limit):
    from datasets import load_dataset

    ds = load_dataset(DATASET_URI, config, split=split, streaming=True)
    n = 0
    for row in ds:
        if row.get("id") in done_ids:
            continue
        yield row
        n += 1
        if limit and n >= limit:
            break


async def run(args):
    _load_dotenv()
    # Size the thread pool to the requested concurrency so it is not silently
    # capped by cpu_count (the default asyncio executor is min(32, cpu+4)).
    from concurrent.futures import ThreadPoolExecutor

    asyncio.get_running_loop().set_default_executor(
        ThreadPoolExecutor(max_workers=args.concurrency + 4)
    )
    if not os.getenv("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY not set (needed for the Claude extractor).")
    keys = [
        k
        for k in (
            os.getenv("OPENROUTER_DEEPSEEK_API_KEY"),
            os.getenv("OPENROUTER_QWEN_API_KEY"),
        )
        if k
    ]
    if not keys:
        sys.exit("No OPENROUTER_* key set (needed for the deepseek judge).")
    print(f"OpenRouter keys available for judge: {len(keys)}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = load_done_ids(out_path)
    print(f"Already done (successful) ids: {len(done)}")

    n_pool = args.concurrency
    print(f"Building {n_pool} pipeline instances ...")
    pool = asyncio.Queue()
    pipes = [make_pipeline(i, keys) for i in range(n_pool)]
    for p in pipes:
        pool.put_nowait(p)

    write_lock = asyncio.Lock()
    counters = {"ok": 0, "err": 0}
    t0 = time.time()

    async def worker(row):
        pipe = await pool.get()
        try:
            res = await asyncio.to_thread(
                process_paper, row, pipe, args.si_char_cap, args.max_materials
            )
        finally:
            pool.put_nowait(pipe)
        async with write_lock:
            with open(out_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(res, ensure_ascii=False) + "\n")
            counters["ok" if res["error"] is None else "err"] += 1
            done_n = counters["ok"] + counters["err"]
            if done_n % 10 == 0 or done_n <= 3:
                rate = done_n / max(time.time() - t0, 1e-9)
                print(
                    f"  {done_n} processed "
                    f"(ok={counters['ok']} err={counters['err']}) "
                    f"{rate:.2f} papers/s"
                )
        return res

    rows = list(iter_rows(args.config, args.split, done, args.limit))
    print(f"Papers to process this run: {len(rows)}")
    if not rows:
        print("Nothing to do.")
        return

    tasks = [asyncio.create_task(worker(r)) for r in rows]
    for coro in asyncio.as_completed(tasks):
        await coro

    # cost report from the tracked LMs
    cost = 0.0
    for p in pipes:
        for comp in ("material", "synthesis", "judge"):
            lm = getattr(p[comp], "lm", None)
            if lm is not None and hasattr(lm, "get_cost"):
                cost += lm.get_cost()
    dt = time.time() - t0
    print(
        f"\nDone: {counters['ok']} ok, {counters['err']} errors in {dt:.0f}s "
        f"({len(rows) / max(dt, 1e-9):.2f} papers/s). "
        f"Tracked LM cost this run: ${cost:.2f}"
    )
    print(f"Results appended to {out_path}")

    if args.push:
        push_to_hub(out_path, args.hub_repo, args.config, args.split)


def push_to_hub(out_path, hub_repo, config, split):
    if not hub_repo:
        sys.exit("--push requires --hub-repo <namespace/name>")
    from datasets import Dataset

    rows = [
        json.loads(line)
        for line in out_path.read_text().splitlines()
        if line.strip()
    ]
    ds = Dataset.from_list(rows)
    print(f"Pushing {len(rows)} rows to {hub_repo} (config={config}, "
          f"split={split}) ...")
    ds.push_to_hub(hub_repo, config_name=config, split=split, private=True)
    print("Pushed. (Private repo; a NEW dataset, source is untouched.)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="superconductor_keywords_and_LLM")
    ap.add_argument("--split", default="full")
    ap.add_argument("--limit", type=int, default=None,
                    help="max papers this run (for smoke tests)")
    ap.add_argument("--concurrency", type=int, default=16,
                    help="pool size == max papers in flight")
    ap.add_argument("--si-char-cap", type=int, default=None,
                    help="optional: truncate text_si to this many chars. "
                         "Default None = use the full SI (no truncation)")
    ap.add_argument("--max-materials", type=int, default=20,
                    help="cap materials per paper to bound cost")
    ap.add_argument("--out", default=None,
                    help="JSONL output path (default under results/)")
    ap.add_argument("--push", action="store_true",
                    help="push accumulated JSONL to a NEW hub dataset repo")
    ap.add_argument("--hub-repo", default=None,
                    help="target repo for --push, e.g. user/LeMat-recipes")
    args = ap.parse_args()
    if args.out is None:
        args.out = f"results/{args.config}_{args.split}.jsonl"
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
