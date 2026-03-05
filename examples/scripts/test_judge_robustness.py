"""
Robustness test for GeneralSynthesisEvaluationScore.overall_score / overall_reasoning
optional fields fix.

Uses Claude as synthesizer (reliable) and Qwen3.5-397b as judge across 4 papers
to verify the judge never fails due to missing overall_score / overall_reasoning.
"""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root / "src"))
os.chdir(repo_root)

from dotenv import load_dotenv
load_dotenv(repo_root / ".env", override=True)

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
from llm_synthesis.utils.concurrency import get_max_concurrent_llm_calls, run_with_semaphore
from llm_synthesis.utils.dspy_utils import get_llm_from_name

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logging.getLogger("LiteLLM").setLevel(logging.ERROR)
logging.getLogger("litellm").setLevel(logging.ERROR)
logging.getLogger("pydantic").setLevel(logging.ERROR)
log = logging.getLogger(__name__)

SYNTH_LLM = "claude-sonnet-4.6"
JUDGE_LLMS = ["claude-sonnet-4.6", "gemini-3-flash", "qwen3.5-397b-a17b", "deepseek-v3.2"]

PAPER_IDS = [
    "2404.08872",                                      # MoS2-rGO hydrothermal
    "3325ac6dfb049a5efdaad7f876c1d51b17be0158",       # Fe3O4 hollow spheres
    "3b87159630bb0581024897961bb5fc922fc3db19",        # MgB2/BN nanotubes
    "9a889c1a671fd3cae48285eaa95069d189d02fe3",        # noble metal nanoparticles
]

MAT_PROMPT_PATH = str(repo_root / "examples/system_prompts/material_extraction/default.txt")
SYNTH_PROMPT_PATH = str(repo_root / "examples/system_prompts/synthesis_extraction/default.txt")
JUDGE_PROMPT_PATH = str(repo_root / "examples/system_prompts/judge/default.txt")

MODEL_KWARGS = {"temperature": 0.0, "max_tokens": 16000, "num_retries": 3, "cache": False}


def build_components():
    with open(MAT_PROMPT_PATH) as f:
        mat_prompt = f.read()
    with open(SYNTH_PROMPT_PATH) as f:
        synth_prompt = f.read()
    with open(JUDGE_PROMPT_PATH) as f:
        judge_prompt = f.read()

    mat_sig = make_dspy_text_extractor_signature(
        signature_name="TextToMaterials",
        instructions="Extract ONLY the final synthesized materials from the publication text.",
        input_description="The publication text to extract the final synthesized materials from.",
        output_name="materials",
        output_description=(
            "The final synthesized materials as a comma-separated list using ONLY "
            "chemical formulas (e.g., 'Sr3(P3O9)2·7H2O, LiH2PO3'). Never use common names."
        ),
    )
    synth_sig = make_dspy_synthesis_extractor_signature(
        signature_name="SynthesisSignature",
        instructions="Extract the structured synthesis for a specific material from the paper text.",
        paper_text_description="The complete paper text to search for the material's synthesis procedure.",
        material_name_description="The name of the specific material to extract synthesis for.",
        output_name="structured_synthesis",
        output_description="The extracted structured synthesis for the specific material.",
    )
    judge_sig = make_general_synthesis_judge_signature(
        signature_name="GeneralSynthesisJudgeSignature",
        instructions=(
            "You are an expert materials scientist. Evaluate how well the "
            "GeneralSynthesisOntology extraction captures synthesis information from "
            "the provided source text. Score 1-5 for each criterion."
        ),
    )

    mat_lm = get_llm_from_name(SYNTH_LLM, model_kwargs=dict(MODEL_KWARGS, max_tokens=4096))
    synth_lm = get_llm_from_name(SYNTH_LLM, model_kwargs=dict(MODEL_KWARGS))
    mat_extractor = DspyTextExtractor(signature=mat_sig, lm=mat_lm)
    synth_extractor = DspySynthesisExtractor(signature=synth_sig, lm=synth_lm)
    judges = {
        name: DspyGeneralSynthesisJudge(
            signature=judge_sig,
            lm=get_llm_from_name(name, model_kwargs=dict(MODEL_KWARGS)),
            enable_reasoning_traces=True,
            confidence_threshold=0.7,
        )
        for name in JUDGE_LLMS
    }
    return mat_extractor, synth_extractor, judges


async def process_paper(paper_id, paper_text, mat_extractor, synth_extractor, judges, semaphore):
    log.info(f"\n[{paper_id}] Starting...")
    no_mat = {"no material", "none", "n/a", "not found", "no synthesis"}

    # Material extraction
    raw = await run_with_semaphore(semaphore, mat_extractor.forward, input=clean_text(paper_text))
    raw = (raw or "").strip()
    if not raw or any(p in raw.lower() for p in no_mat):
        log.warning(f"[{paper_id}] No materials found")
        return {"paper_id": paper_id, "materials": []}

    materials = [m.strip() for m in raw.replace("\n", ",").split(",") if m.strip()]
    log.info(f"[{paper_id}] Materials: {materials}")

    results = []
    for material in materials:
        # Synthesis extraction (Claude)
        try:
            synthesis = await run_with_semaphore(
                semaphore, synth_extractor.forward, input=(clean_text(paper_text), material)
            )
        except Exception as e:
            log.error(f"[{paper_id}] Synthesis failed for {material}: {e}")
            synthesis = GeneralSynthesisOntology(
                target_compound=material, target_compound_type="other",
                synthesis_method="other", starting_materials=[], steps=[], equipment=[],
                notes=f"Failed: {e}",
            )

        # All 4 judges in parallel
        judge_input = (clean_text(paper_text), json.dumps(synthesis.model_dump()), material)
        judge_results = await asyncio.gather(*[
            run_with_semaphore(semaphore, judges[jn].forward, judge_input)
            for jn in JUDGE_LLMS
        ], return_exceptions=True)

        row = {"material": material, "scores": {}}
        for jn, jr in zip(JUDGE_LLMS, judge_results):
            if isinstance(jr, Exception):
                log.error(f"[{paper_id}] [{jn}] Judge FAILED for {material}: {jr}")
                row["scores"][jn] = {"score": None, "error": str(jr)}
            else:
                score = jr.scores.overall_score
                reasoning = jr.scores.overall_reasoning
                log.info(f"[{paper_id}] [{jn}] {material}: score={score} reasoning={'yes' if reasoning else 'no'}")
                row["scores"][jn] = {"score": score, "error": None}
        results.append(row)

    return {"paper_id": paper_id, "materials": results}


async def main():
    papers = json.load(open(repo_root / "examples/data/annotation_papers.json"))
    paper_map = {p["id"]: p for p in papers}

    mat_extractor, synth_extractor, judges = build_components()
    semaphore = asyncio.Semaphore(get_max_concurrent_llm_calls())

    tasks = [
        process_paper(
            pid,
            paper_map[pid]["text_paper"],
            mat_extractor, synth_extractor, judges, semaphore,
        )
        for pid in PAPER_IDS
        if pid in paper_map and paper_map[pid]["text_paper"]
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    log.info("\n" + "=" * 60)
    log.info("ROBUSTNESS SUMMARY")
    log.info("=" * 60)

    # Header
    header = f"  {'Paper':12} {'Material':20} " + " ".join(f"{jn:20}" for jn in JUDGE_LLMS)
    log.info(header)
    log.info("-" * len(header))

    total = {jn: 0 for jn in JUDGE_LLMS}
    failures = {jn: 0 for jn in JUDGE_LLMS}
    for r in results:
        if isinstance(r, Exception):
            log.error(f"Paper task failed: {r}")
            continue
        for m in r["materials"]:
            scores_str = ""
            for jn in JUDGE_LLMS:
                total[jn] += 1
                s = m["scores"].get(jn, {})
                if s.get("error") or s.get("score") is None:
                    failures[jn] += 1
                    scores_str += f"{'FAIL':20}"
                else:
                    scores_str += f"{str(s['score']):20}"
            log.info(f"  {r['paper_id'][:12]:12} {m['material']:20} {scores_str}")
    log.info("")
    for jn in JUDGE_LLMS:
        ok = total[jn] - failures[jn]
        log.info(f"  {jn}: {ok}/{total[jn]} succeeded")


if __name__ == "__main__":
    asyncio.run(main())
