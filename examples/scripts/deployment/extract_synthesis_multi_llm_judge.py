# This script is based on extract_synthesis_procedure_from_text.py
# The logic from there is extended to run mxn synthesis x evalutions
# and generates a result.json and evaluation matrix summarizing the results.

import asyncio
import json
import logging
import os
import random
import warnings

import matplotlib

matplotlib.use("Agg")
import dspy
import hydra
import numpy as np
from hydra.utils import get_original_cwd, instantiate
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from omegaconf import DictConfig, OmegaConf

from llm_synthesis.data_loader.paper_loader.base import PaperLoaderInterface
from llm_synthesis.metrics.judge.general_synthesis_judge import (
    DspyGeneralSynthesisJudge,
)
from llm_synthesis.models.ontologies.general import GeneralSynthesisOntology
from llm_synthesis.transformers.material_extraction.base import (
    MaterialExtractorInterface,
)
from llm_synthesis.transformers.synthesis_extraction.base import (
    SynthesisExtractorInterface,
)
from llm_synthesis.utils import clean_text
from llm_synthesis.utils.concurrency import (
    get_max_concurrent_llm_calls,
    run_with_semaphore,
)
from llm_synthesis.utils.dspy_utils import get_lm_cost

# Disable Pydantic warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

# Configure logging to reduce noise
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
)
logging.getLogger("pydantic").setLevel(logging.ERROR)
logging.getLogger("LiteLLM").setLevel(logging.ERROR)
logging.getLogger("litellm").setLevel(logging.ERROR)


def _resolve_prompt_path(cfg_section, original_cwd: str):
    """Resolve system prompt path in-place, same as the original script."""
    if hasattr(cfg_section.architecture.lm.system_prompt, "prompt_path"):
        cfg_section.architecture.lm.system_prompt.prompt_path = os.path.join(
            original_cwd,
            cfg_section.architecture.lm.system_prompt.prompt_path,
        )


def _build_component(cfg_section, llm_name: str):
    """Instantiate a component with a specific LLM"""
    OmegaConf.set_struct(cfg_section.architecture.lm, False)
    cfg_section.architecture.lm.llm_name = llm_name
    return instantiate(cfg_section.architecture)


def _save_matrix_png(summary, synthesis_llms, judge_llms, title, output_path):
    """Save a heatmap PNG of the evaluation matrix (thread-safe)."""
    data = np.full((len(synthesis_llms), len(judge_llms)), np.nan)
    for i, s_llm in enumerate(synthesis_llms):
        for j, j_llm in enumerate(judge_llms):
            val = summary.get(s_llm, {}).get(j_llm, {}).get("avg_overall_score")
            if val is not None:
                data[i, j] = val

    masked_data = np.ma.masked_invalid(data)
    cmap = matplotlib.colormaps["RdYlGn"].copy()
    cmap.set_bad(color="#d9d9d9")  # for n/a values

    fig = Figure(
        figsize=(max(5, len(judge_llms) * 2), max(4, len(synthesis_llms) * 1.5))
    )
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    im = ax.imshow(masked_data, vmin=1.0, vmax=5.0, cmap=cmap, aspect="auto")

    ax.set_xticks(range(len(judge_llms)))
    ax.set_yticks(range(len(synthesis_llms)))
    ax.set_xticklabels(judge_llms, fontsize=10)
    ax.set_yticklabels(synthesis_llms, fontsize=10)
    ax.set_xlabel("Judge LLM", fontsize=12, labelpad=8)
    ax.set_ylabel("Synthesis LLM", fontsize=12, labelpad=8)
    ax.set_title(title, fontsize=13, pad=12)

    for i in range(len(synthesis_llms)):
        for j in range(len(judge_llms)):
            val = data[i, j]
            text = f"{val:.2f}" if not np.isnan(val) else "N/A"
            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                fontsize=12,
                fontweight="bold",
                color="black",
            )

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Avg Score (1-5)")
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    del canvas


@hydra.main(
    config_path="../../config", config_name="config.yaml", version_base=None
)
def main(cfg: DictConfig) -> None:
    original_cwd = get_original_cwd()

    # Ensure data directory is correctly set if it's defined in the config
    if hasattr(cfg.data_loader.architecture, "data_dir"):
        if not (
            cfg.data_loader.architecture.data_dir.startswith("s3://")
            or cfg.data_loader.architecture.data_dir.startswith("gs://")
            or cfg.data_loader.architecture.data_dir.startswith("/")
        ):
            cfg.data_loader.architecture.data_dir = os.path.join(
                original_cwd, cfg.data_loader.architecture.data_dir
            )

    if hasattr(cfg.data_loader.architecture, "annotations_dir"):
        if not cfg.data_loader.architecture.annotations_dir.startswith("/"):
            cfg.data_loader.architecture.annotations_dir = os.path.join(
                original_cwd, cfg.data_loader.architecture.annotations_dir
            )

    # Load data
    data_loader: PaperLoaderInterface = instantiate(
        cfg.data_loader.architecture
    )
    papers = data_loader.load()

    # if the key cfg.data_loader.number_of_samples is set, take n random samples
    if cfg.data_loader.number_of_samples:
        papers = random.sample(papers, cfg.data_loader.number_of_samples)

    # Handle system prompt paths if defined
    _resolve_prompt_path(cfg.material_extraction, original_cwd)
    _resolve_prompt_path(cfg.synthesis_extraction, original_cwd)
    _resolve_prompt_path(cfg.judge, original_cwd)

    synthesis_llms = list(cfg.synthesis_extraction.llm_names)
    judge_llms = list(cfg.judge.llm_names)

    logging.info(f"Synthesis LLMs (m={len(synthesis_llms)}): {synthesis_llms}")
    logging.info(f"Judge LLMs (n={len(judge_llms)}): {judge_llms}")

    # Build components
    # synthesis_llms drives both material + synthesis extraction (same LLM/pair)
    mat_extractors: dict[str, MaterialExtractorInterface] = {
        name: _build_component(cfg.material_extraction, name)
        for name in synthesis_llms
    }
    synthesis_extractors: dict[str, SynthesisExtractorInterface] = {
        name: _build_component(cfg.synthesis_extraction, name)
        for name in synthesis_llms
    }
    judges: dict[str, DspyGeneralSynthesisJudge] = {
        name: _build_component(cfg.judge, name) for name in judge_llms
    }

    # Result gatherer (use result_save=multi_llm config)
    result_gather = instantiate(cfg.result_save.architecture)
    result_dir = cfg.result_save.architecture.result_dir

    # LM refs for per-operation cost tracking
    synthesis_lms = {
        name: getattr(synthesis_extractors[name], "lm", None)
        for name in synthesis_llms
    }
    judge_lms = {name: getattr(judges[name], "lm", None) for name in judge_llms}
    dspy_settings_lm = getattr(dspy.settings, "lm", None)

    # Papers to process (skip already-processed)
    to_process = [p for p in papers if p.id not in os.listdir(result_dir)]

    if cfg.data_loader.number_of_samples:
        to_process = random.sample(
            to_process, cfg.data_loader.number_of_samples
        )

    total_cost = 0.0
    max_concurrent_llm = get_max_concurrent_llm_calls()
    llm_semaphore = asyncio.Semaphore(max_concurrent_llm)
    logging.info(f"Max concurrent LLM calls: {max_concurrent_llm}")

    async def process_paper_async(paper) -> tuple:
        """Process a single paper with concurrent LLM calls.

        Returns:
            (summary, cost).
        """
        logging.info(f"Processing {paper.name}")
        multi_llm_results = []
        eval_matrix = {}
        cost_operations = []

        initial_dspy_cost = (
            get_lm_cost(dspy_settings_lm) if dspy_settings_lm else 0.0
        )

        try:
            # --- Material extraction: parallel across synthesis_llms ---
            initial_synth_costs = {
                s: get_lm_cost(synthesis_lms.get(s)) or 0.0
                for s in synthesis_llms
            }
            material_texts = await asyncio.gather(
                *[
                    run_with_semaphore(
                        llm_semaphore,
                        mat_extractors[synth_llm].forward,
                        input=clean_text(paper.publication_text),
                    )
                    for synth_llm in synthesis_llms
                ]
            )
            for synth_llm, materials_text in zip(
                synthesis_llms, material_texts
            ):
                synth_lm = synthesis_lms.get(synth_llm)
                cost_operations.append(
                    {
                        "operation": "material_extraction",
                        "synth_llm": synth_llm,
                        "cost_usd": (get_lm_cost(synth_lm) or 0.0)
                        - initial_synth_costs[synth_llm],
                    }
                )

            _no_mat_phrases = {
                "no material",
                "none",
                "n/a",
                "not found",
                "no synthesis",
            }
            materials_per_llm = {}
            for synth_llm, materials_text in zip(
                synthesis_llms, material_texts
            ):
                raw = (materials_text or "").strip()
                if any(p in raw.lower() for p in _no_mat_phrases):
                    materials_per_llm[synth_llm] = []
                else:
                    materials_per_llm[synth_llm] = [
                        m.strip()
                        for m in (materials_text or "")
                        .replace("\n", ",")
                        .split(",")
                        if m.strip()
                    ]

            # --- Synthesis extraction + judge evaluation: across all pairs ---
            async def process_pair(synth_llm: str, material: str):
                """
                Run synthesis extraction then judge evaluation for one pair.
                """
                synth_lm = synthesis_lms.get(synth_llm)
                pair_cost_ops = []

                cost_before_synth = get_lm_cost(synth_lm) if synth_lm else 0.0
                logging.info(f"  [{synth_llm}] Synthesis -> {material}")
                try:
                    synthesis = await run_with_semaphore(
                        llm_semaphore,
                        synthesis_extractors[synth_llm].forward,
                        input=(clean_text(paper.publication_text), material),
                    )
                except Exception as e:
                    logging.error(f"Synthesis failed for {material}: {e}")
                    synthesis = GeneralSynthesisOntology(
                        target_compound=material,
                        target_compound_type="other",
                        synthesis_method="other",
                        starting_materials=[],
                        steps=[],
                        equipment=[],
                        notes=f"Processing failed: {e!s}",
                    )
                cost_after_synth = get_lm_cost(synth_lm) if synth_lm else 0.0
                pair_cost_ops.append(
                    {
                        "operation": "synthesis_extraction",
                        "synth_llm": synth_llm,
                        "material": material,
                        "cost_usd": cost_after_synth - cost_before_synth,
                    }
                )

                cost_before_judges = {
                    j: get_lm_cost(judge_lms.get(j)) or 0.0 for j in judge_llms
                }
                judge_input = (
                    clean_text(paper.publication_text),
                    json.dumps(synthesis.model_dump()),
                    material,
                )
                judge_results = await asyncio.gather(
                    *[
                        run_with_semaphore(
                            llm_semaphore,
                            judges[judge_llm].forward,
                            judge_input,
                        )
                        for judge_llm in judge_llms
                    ],
                    return_exceptions=True,
                )
                cost_after_judges = {
                    j: get_lm_cost(judge_lms.get(j)) or 0.0 for j in judge_llms
                }
                for judge_llm in judge_llms:
                    pair_cost_ops.append(
                        {
                            "operation": "evaluation",
                            "synth_llm": synth_llm,
                            "judge_llm": judge_llm,
                            "material": material,
                            "cost_usd": cost_after_judges[judge_llm]
                            - cost_before_judges[judge_llm],
                        }
                    )

                return (
                    synth_llm, material, synthesis, judge_results, pair_cost_ops
                )

            # Handle synth_llms with no materials; log materials found
            for synth_llm in synthesis_llms:
                materials = materials_per_llm.get(synth_llm, [])
                if not materials:
                    logging.warning(
                        f"No materials found for paper {paper.name} "
                        f"with llm {synth_llm}"
                    )
                    multi_llm_results.append(
                        {
                            "synth_llm": synth_llm,
                            "materials": [],
                            "note": "No materials found",
                        }
                    )
                else:
                    logging.info(f"[{synth_llm}] Found materials: {materials}")

            all_pairs = [
                (synth_llm, material)
                for synth_llm in synthesis_llms
                for material in materials_per_llm.get(synth_llm, [])
            ]
            if all_pairs:
                pair_results = await asyncio.gather(
                    *[process_pair(sl, m) for sl, m in all_pairs],
                    return_exceptions=True,
                )

                # Assemble results in synthesis_llms order
                synth_entries: dict[str, dict] = {}
                for item in pair_results:
                    if isinstance(item, Exception):
                        logging.error(f"Pair task failed: {item}")
                        continue
                    (
                        synth_llm, material, synthesis, jresults, pair_cost_ops
                    ) = item
                    cost_operations.extend(pair_cost_ops)

                    if synth_llm not in synth_entries:
                        synth_entries[synth_llm] = {
                            "synth_llm": synth_llm,
                            "materials": [],
                        }
                        eval_matrix[synth_llm] = {}

                    evaluations = []
                    for judge_llm, result in zip(judge_llms, jresults):
                        if judge_llm not in eval_matrix[synth_llm]:
                            eval_matrix[synth_llm][judge_llm] = {}
                        if isinstance(result, Exception):
                            logging.error(
                                f"Evaluation failed for {material} "
                                f"({judge_llm}): {result}"
                            )
                            eval_matrix[synth_llm][judge_llm][material] = None
                            evaluations.append(
                                {
                                    "judge_llm": judge_llm,
                                    "evaluation": None,
                                    "overall_score": None,
                                }
                            )
                        else:
                            score = result.scores.overall_score
                            logging.info(
                                f"    Score [{judge_llm}]: {score}/5.0"
                            )
                            eval_matrix[synth_llm][judge_llm][material] = score
                            evaluations.append(
                                {
                                    "judge_llm": judge_llm,
                                    "evaluation": result.model_dump(),
                                    "overall_score": score,
                                }
                            )

                    synth_entries[synth_llm]["materials"].append(
                        {
                            "material": material,
                            "synthesis": synthesis.model_dump(),
                            "evaluations": evaluations,
                        }
                    )

                for synth_llm in synthesis_llms:
                    if synth_llm in synth_entries:
                        multi_llm_results.append(synth_entries[synth_llm])

            # Build summary for heatmap
            summary = {}
            for synth_llm in synthesis_llms:
                summary[synth_llm] = {}
                for judge_llm in judge_llms:
                    scores = eval_matrix.get(synth_llm, {}).get(judge_llm, {})
                    valid = [s for s in scores.values() if s is not None]
                    avg = round(sum(valid) / len(valid), 2) if valid else None
                    summary[synth_llm][judge_llm] = {
                        "avg_overall_score": avg,
                        "num_materials": len(valid),
                    }

            final_dspy_cost = (
                get_lm_cost(dspy_settings_lm) if dspy_settings_lm else 0.0
            )
            dspy_cost = (final_dspy_cost or 0.0) - (initial_dspy_cost or 0.0)
            if dspy_cost > 0:
                cost_operations.append(
                    {
                        "operation": "dspy_settings_lm",
                        "cost_usd": dspy_cost,
                    }
                )

            paper_cost = sum(op["cost_usd"] for op in cost_operations)

            result_gather.gather(
                paper_id=paper.id,
                publication_text=paper.publication_text,
                si_text=paper.si_text,
                multi_llm_results=multi_llm_results,
                cost_data=cost_operations,
            )

            paper_dir = os.path.join(result_dir, paper.id)
            heatmap_data = {
                s: {j: v.get("avg_overall_score") for j, v in jd.items()}
                for s, jd in summary.items()
            }
            logging.info(
                f"  Heatmap summary for {paper.name}: "
                f"{json.dumps(heatmap_data)}"
            )
            _save_matrix_png(
                summary,
                synthesis_llms,
                judge_llms,
                title=f"Evaluation Matrix - {paper.name}",
                output_path=os.path.join(paper_dir, "evaluation_matrix.png"),
            )

            num_materials = sum(len(e["materials"]) for e in multi_llm_results)
            logging.info(
                f"Processed {num_materials} material entries across "
                f"{len(multi_llm_results)} LLMs"
            )
            logging.info(f"Paper cost: ${paper_cost:.6f}")

            return summary, paper_cost

        except Exception as e:
            logging.error(f"Failed to process paper {paper.name}: {e}")
            return None, 0.0

    # Cap concurrent papers (same as old max_workers=4)
    max_concurrent_papers = 4
    paper_semaphore = asyncio.Semaphore(max_concurrent_papers)

    async def run_one_paper(paper):
        async with paper_semaphore:
            return paper, await process_paper_async(paper)

    async def run_all_papers():
        all_paper_results = []
        nonlocal total_cost
        if not to_process:
            return all_paper_results
        results = await asyncio.gather(
            *[run_one_paper(paper) for paper in to_process],
            return_exceptions=True,
        )
        for item in results:
            if isinstance(item, Exception):
                logging.error(f"Paper task failed: {item}")
                continue
            paper, (summary, cost) = item
            if summary is not None:
                all_paper_results.append(summary)
                total_cost += cost
                logging.info(f"Finished {paper.name}: cost=${cost:.6f}")
        return all_paper_results

    logging.info(
        f"Processing {len(to_process)} papers "
        f"(max {max_concurrent_papers} papers, {max_concurrent_llm} LLM calls)"
    )
    all_paper_results = asyncio.run(run_all_papers())

    # Save global evaluation matrix PNG
    if all_paper_results:
        totals = {s: {j: [] for j in judge_llms} for s in synthesis_llms}
        for pr in all_paper_results:
            for s in synthesis_llms:
                for j in judge_llms:
                    val = pr.get(s, {}).get(j, {}).get("avg_overall_score")
                    if val is not None:
                        totals[s][j].append(val)
        global_summary = {
            s: {
                j: {
                    "avg_overall_score": round(sum(v) / len(v), 2)
                    if v
                    else None
                }
                for j, v in jd.items()
            }
            for s, jd in totals.items()
        }
        _save_matrix_png(
            global_summary,
            synthesis_llms,
            judge_llms,
            title=(
                f"Evaluation Matrix (avg over {len(all_paper_results)} papers)"
            ),
            output_path=os.path.join(
                result_dir, "global_avg_evaluation_matrix.png"
            ),
        )

    logging.info(f"Total cost across all papers: ${total_cost:.6f}")
    logging.info("Success")


if __name__ == "__main__":
    main()
