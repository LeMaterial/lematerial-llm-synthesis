"""
End-to-end test for both single-LLM and multi-LLM pipelines.

Loads paper 2404.08872 (MoS2-rGO hydrothermal synthesis) from
annotation_papers.json and runs:
  1. Single-LLM mode: claude-sonnet-4.6 for all steps
  2. Multi-LLM mode: 4 models x 4 judges

Outputs saved to examples/data/test_outputs/
"""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from llm_synthesis.metrics.judge.general_synthesis_judge import (
    DspyGeneralSynthesisJudge,
    make_general_synthesis_judge_signature,
)
from llm_synthesis.models.ontologies.general import GeneralSynthesisOntology
from llm_synthesis.transformers.material_extraction.dspy_extraction import (
    DspyTextExtractor,
    make_dspy_text_extractor_signature,
)
from llm_synthesis.transformers.synthesis_extraction.dspy_synthesis_extraction import (
    DspySynthesisExtractor,
    make_dspy_synthesis_extractor_signature,
)
from llm_synthesis.utils import clean_text
from llm_synthesis.utils.concurrency import (
    get_max_concurrent_llm_calls,
    run_with_semaphore,
)
from llm_synthesis.utils.dspy_utils import get_llm_from_name

# --- path setup ---
repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root / "src"))
os.chdir(repo_root)
load_dotenv(repo_root / ".env", override=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logging.getLogger("LiteLLM").setLevel(logging.ERROR)
logging.getLogger("litellm").setLevel(logging.ERROR)
logging.getLogger("pydantic").setLevel(logging.ERROR)
log = logging.getLogger(__name__)

PAPER_IDS = [
    "2404.08872",                                      # MoS2-rGO hydrothermal
    "3325ac6dfb049a5efdaad7f876c1d51b17be0158",       # Fe3O4 hollow spheres
    "3b87159630bb0581024897961bb5fc922fc3db19",        # MgB2/BN nanotubes
    "9a889c1a671fd3cae48285eaa95069d189d02fe3",        # noble metal nanoparticles
]
OUTPUT_BASE_DIR = repo_root / "examples" / "data" / "test_outputs" / "multi_paper"

SYNTH_PROMPT_PATH = str(repo_root / "examples/system_prompts/synthesis_extraction/default.txt")
MAT_PROMPT_PATH = str(repo_root / "examples/system_prompts/material_extraction/default.txt")
JUDGE_PROMPT_PATH = str(repo_root / "examples/system_prompts/judge/default.txt")

SYNTHESIS_LLMS = ["claude-sonnet-4.6", "gemini-3-flash", "qwen3.5-397b-a17b", "deepseek-v3.2"]
JUDGE_LLMS = ["claude-sonnet-4.6", "gemini-3-flash", "qwen3.5-397b-a17b", "deepseek-v3.2"]

MODEL_KWARGS_MAT = {"temperature": 0.0, "max_tokens": 4096, "num_retries": 3, "cache": False}
MODEL_KWARGS_SYNTH = {"temperature": 0.0, "max_tokens": 12000, "num_retries": 3, "cache": False}
MODEL_KWARGS_JUDGE = {"temperature": 0.0, "max_tokens": 16000, "num_retries": 3, "cache": False}


def load_prompts():
    with open(MAT_PROMPT_PATH) as f:
        mat_prompt = f.read()
    with open(SYNTH_PROMPT_PATH) as f:
        synth_prompt = f.read()
    with open(JUDGE_PROMPT_PATH) as f:
        judge_prompt = f.read()
    return mat_prompt, synth_prompt, judge_prompt


def build_mat_extractor(llm_name: str, system_prompt: str) -> DspyTextExtractor:
    sig = make_dspy_text_extractor_signature(
        signature_name="TextToMaterials",
        instructions="Extract ONLY the final synthesized materials from the publication text.",
        input_description="The publication text to extract the final synthesized materials from.",
        output_name="materials",
        output_description=(
            "The final synthesized materials as a comma-separated list using ONLY "
            "chemical formulas (e.g., 'Sr3(P3O9)2·7H2O, LiH2PO3'). Never use common names."
        ),
    )
    lm = get_llm_from_name(llm_name, model_kwargs=dict(MODEL_KWARGS_MAT), system_prompt=system_prompt)
    return DspyTextExtractor(signature=sig, lm=lm)


def build_synth_extractor(llm_name: str, system_prompt: str) -> DspySynthesisExtractor:
    sig = make_dspy_synthesis_extractor_signature(
        signature_name="SynthesisSignature",
        instructions="Extract the structured synthesis for a specific material from the paper text.",
        paper_text_description="The complete paper text to search for the material's synthesis procedure.",
        material_name_description="The name of the specific material to extract synthesis for.",
        output_name="structured_synthesis",
        output_description="The extracted structured synthesis for the specific material.",
    )
    lm = get_llm_from_name(llm_name, model_kwargs=dict(MODEL_KWARGS_SYNTH), system_prompt=system_prompt)
    return DspySynthesisExtractor(signature=sig, lm=lm)


def build_judge(llm_name: str, system_prompt: str) -> DspyGeneralSynthesisJudge:
    sig = make_general_synthesis_judge_signature(
        signature_name="GeneralSynthesisJudgeSignature",
        instructions=(
            "You are an expert materials scientist. Evaluate how well the "
            "GeneralSynthesisOntology extraction captures synthesis information from "
            "the provided source text. Score 1-5 for each criterion."
        ),
    )
    lm = get_llm_from_name(llm_name, model_kwargs=dict(MODEL_KWARGS_JUDGE), system_prompt=system_prompt)
    return DspyGeneralSynthesisJudge(
        signature=sig, lm=lm, enable_reasoning_traces=True, confidence_threshold=0.7,
    )


# ── Single-LLM test ────────────────────────────────────────────────────────────

def run_single_llm(paper_text: str, llm_name: str, mat_prompt: str, synth_prompt: str, judge_prompt: str) -> dict:
    log.info(f"\n{'='*60}\nSINGLE-LLM TEST: {llm_name}\n{'='*60}")
    result = {"llm": llm_name, "materials": []}

    mat_extractor = build_mat_extractor(llm_name, mat_prompt)
    synth_extractor = build_synth_extractor(llm_name, synth_prompt)
    judge = build_judge(llm_name, judge_prompt)

    # Material extraction
    log.info("Step 1: Material extraction...")
    materials_text = mat_extractor.forward(input=clean_text(paper_text))
    log.info(f"  Raw output: {materials_text!r}")

    no_mat = {"no material", "none", "n/a", "not found", "no synthesis"}
    if not materials_text or any(p in (materials_text or "").lower() for p in no_mat):
        log.warning("  No materials found")
        return result

    materials = [m.strip() for m in (materials_text or "").replace("\n", ",").split(",") if m.strip()]
    log.info(f"  Materials: {materials}")

    for material in materials:
        log.info(f"\nStep 2: Synthesis extraction for {material}...")
        try:
            synthesis = synth_extractor.forward(input=(clean_text(paper_text), material))
            log.info(f"  Method: {synthesis.synthesis_method}, Steps: {len(synthesis.steps)}")
        except Exception as e:
            log.error(f"  Synthesis failed: {e}")
            synthesis = GeneralSynthesisOntology(
                target_compound=material, target_compound_type="other",
                synthesis_method="other", starting_materials=[], steps=[], equipment=[],
                notes=f"Failed: {e}",
            )

        log.info(f"Step 3: Judge evaluation for {material}...")
        try:
            judge_result = judge.forward((
                clean_text(paper_text),
                json.dumps(synthesis.model_dump()),
                material,
            ))
            score = judge_result.scores.overall_score
            log.info(f"  Overall score: {score}/5.0")
        except Exception as e:
            log.error(f"  Judge failed: {e}")
            judge_result = None
            score = None

        result["materials"].append({
            "material": material,
            "synthesis": synthesis.model_dump(),
            "evaluation": judge_result.model_dump() if judge_result else None,
            "overall_score": score,
        })

    return result


# ── Multi-LLM test ─────────────────────────────────────────────────────────────

async def run_multi_llm(paper_text: str, mat_prompt: str, synth_prompt: str, judge_prompt: str) -> dict:
    log.info(f"\n{'='*60}\nMULTI-LLM TEST: {SYNTHESIS_LLMS} x {JUDGE_LLMS}\n{'='*60}")

    semaphore = asyncio.Semaphore(get_max_concurrent_llm_calls())

    mat_extractors = {n: build_mat_extractor(n, mat_prompt) for n in SYNTHESIS_LLMS}
    synth_extractors = {n: build_synth_extractor(n, synth_prompt) for n in SYNTHESIS_LLMS}
    judges = {n: build_judge(n, judge_prompt) for n in JUDGE_LLMS}

    # Step 1: parallel material extraction across all synthesis LLMs
    log.info("Step 1: Parallel material extraction across all synthesis LLMs...")
    mat_results = await asyncio.gather(*[
        run_with_semaphore(semaphore, mat_extractors[n].forward, input=clean_text(paper_text))
        for n in SYNTHESIS_LLMS
    ], return_exceptions=True)

    no_mat = {"no material", "none", "n/a", "not found", "no synthesis"}
    materials_per_llm = {}
    for llm_name, raw in zip(SYNTHESIS_LLMS, mat_results):
        if isinstance(raw, Exception):
            log.error(f"  [{llm_name}] Material extraction failed: {raw}")
            materials_per_llm[llm_name] = []
            continue
        raw = (raw or "").strip()
        if any(p in raw.lower() for p in no_mat):
            materials_per_llm[llm_name] = []
        else:
            materials_per_llm[llm_name] = [m.strip() for m in raw.replace("\n", ",").split(",") if m.strip()]
        log.info(f"  [{llm_name}] Materials: {materials_per_llm[llm_name]}")

    # Step 2+3: parallel synthesis + judge per (synth_llm, material) pair
    async def process_pair(synth_llm: str, material: str):
        log.info(f"  [{synth_llm}] Synthesis -> {material}")
        try:
            synthesis = await run_with_semaphore(
                semaphore, synth_extractors[synth_llm].forward,
                input=(clean_text(paper_text), material),
            )
        except Exception as e:
            log.error(f"  [{synth_llm}] Synthesis failed for {material}: {e}")
            synthesis = GeneralSynthesisOntology(
                target_compound=material, target_compound_type="other",
                synthesis_method="other", starting_materials=[], steps=[], equipment=[],
                notes=f"Failed: {e}",
            )

        judge_input = (clean_text(paper_text), json.dumps(synthesis.model_dump()), material)
        judge_results = await asyncio.gather(*[
            run_with_semaphore(semaphore, judges[jn].forward, judge_input)
            for jn in JUDGE_LLMS
        ], return_exceptions=True)

        evaluations = {}
        for judge_llm, jr in zip(JUDGE_LLMS, judge_results):
            if isinstance(jr, Exception):
                log.error(f"  [{synth_llm}→{judge_llm}] Judge failed for {material}: {jr}")
                evaluations[judge_llm] = {"error": str(jr), "overall_score": None}
            else:
                score = jr.scores.overall_score
                log.info(f"  [{synth_llm}→{judge_llm}] Score: {score}/5.0")
                evaluations[judge_llm] = {"evaluation": jr.model_dump(), "overall_score": score}

        return synth_llm, material, synthesis, evaluations

    all_pairs = [
        (sl, m) for sl in SYNTHESIS_LLMS for m in materials_per_llm.get(sl, [])
    ]
    log.info(f"\nStep 2+3: Processing {len(all_pairs)} (synth_llm, material) pairs in parallel...")
    pair_results = await asyncio.gather(*[process_pair(sl, m) for sl, m in all_pairs], return_exceptions=True)

    # Assemble
    output: dict = {sl: {"materials": []} for sl in SYNTHESIS_LLMS}
    for item in pair_results:
        if isinstance(item, Exception):
            log.error(f"Pair failed: {item}")
            continue
        synth_llm, material, synthesis, evaluations = item
        output[synth_llm]["materials"].append({
            "material": material,
            "synthesis": synthesis.model_dump(),
            "evaluations": evaluations,
        })

    return output


# ── Per-paper runner ───────────────────────────────────────────────────────────

async def run_paper(paper_id, paper_text, paper_title, mat_prompt, synth_prompt, judge_prompt):
    log.info(f"\n{'='*60}\nPAPER: {paper_id}\nTitle: {paper_title}\n{'='*60}")
    out_dir = OUTPUT_BASE_DIR / paper_id
    os.makedirs(out_dir, exist_ok=True)

    # ── Single-LLM: run each synthesis LLM as a self-contained pipeline ──
    # (mat extraction + synthesis extraction + judging all with the same LLM)
    single_results = {}
    for llm_name in SYNTHESIS_LLMS:
        log.info(f"[{paper_id}] Single-LLM: {llm_name}")
        result = await asyncio.to_thread(
            run_single_llm, paper_text, llm_name, mat_prompt, synth_prompt, judge_prompt
        )
        single_results[llm_name] = result
        safe_name = llm_name.replace("/", "-").replace(".", "-")
        single_path = out_dir / f"single_llm_{safe_name}.json"
        with open(single_path, "w") as f:
            json.dump(result, f, indent=2)
        log.info(f"[{paper_id}] Single-LLM [{llm_name}] saved → {single_path}")

    # ── Multi-LLM: all synthesis LLMs × all judge LLMs ──
    multi_result = await run_multi_llm(paper_text, mat_prompt, synth_prompt, judge_prompt)
    multi_path = out_dir / "multi_llm.json"
    with open(multi_path, "w") as f:
        json.dump(multi_result, f, indent=2)
    log.info(f"[{paper_id}] Multi-LLM saved → {multi_path}")

    return paper_id, single_results, multi_result


# ── Main ───────────────────────────────────────────────────────────────────────

async def main_async():
    papers = json.load(open(repo_root / "examples/data/annotation_papers.json"))
    paper_map = {p["id"]: p for p in papers}

    mat_prompt, synth_prompt, judge_prompt = load_prompts()
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

    # Run papers sequentially to avoid overloading APIs.
    # (Each paper's single-LLM and multi-LLM runs are parallelised internally.)
    all_results = []
    for pid in PAPER_IDS:
        if pid not in paper_map or not paper_map[pid].get("text_paper"):
            log.warning(f"Paper {pid} not found or has no text — skipping")
            continue
        p = paper_map[pid]
        result = await run_paper(
            pid, p["text_paper"], p.get("title", pid),
            mat_prompt, synth_prompt, judge_prompt,
        )
        all_results.append(result)

    # ── Combined summary ──
    log.info("\n" + "="*60)
    log.info("MULTI-PAPER SUMMARY")
    log.info("="*60)
    for paper_id, single_results, multi_result in all_results:
        log.info(f"\nPaper: {paper_id}")

        log.info("  Single-LLM scores (each LLM judges its own extraction):")
        for llm_name, result in single_results.items():
            for m in result.get("materials", []):
                log.info(f"    [{llm_name}] {m['material']}: score={m['overall_score']}")

        log.info("  Multi-LLM scores (synthesis LLM → judge LLM):")
        for sl in SYNTHESIS_LLMS:
            for entry in multi_result.get(sl, {}).get("materials", []):
                scores = {
                    jl: entry["evaluations"].get(jl, {}).get("overall_score")
                    for jl in JUDGE_LLMS
                }
                log.info(f"    [{sl}] {entry['material']}: {scores}")

    log.info(f"\nAll outputs saved to: {OUTPUT_BASE_DIR}")


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
