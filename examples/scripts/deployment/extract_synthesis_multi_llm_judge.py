#This script is based on extract_synthesis_procedure_from_text.py
#The logic from there is extended to run mxn synthesis x evalutions
#and generates a result.json and evaluation matrix summarizing the results.

import json
import logging
import os
import random
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import numpy as np

import dspy
import hydra
from hydra.utils import get_original_cwd, instantiate
from omegaconf import DictConfig, OmegaConf

from llm_synthesis.data_loader.paper_loader.base import PaperLoaderInterface
from llm_synthesis.metrics.judge.general_synthesis_judge import DspyGeneralSynthesisJudge
from llm_synthesis.models.ontologies.general import GeneralSynthesisOntology
from llm_synthesis.transformers.material_extraction.base import MaterialExtractorInterface
from llm_synthesis.transformers.synthesis_extraction.base import SynthesisExtractorInterface
from llm_synthesis.utils import clean_text
from llm_synthesis.utils.dspy_utils import get_lm_cost

# Disable Pydantic warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

# Configure logging to reduce noise
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
    """Instantiate a component with a specific LLM """
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
    cmap.set_bad(color="#d9d9d9")         #for n/a values

    fig = Figure(figsize=(max(5, len(judge_llms) * 2), max(4, len(synthesis_llms) * 1.5)))
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
            ax.text(j, i, text, ha="center", va="center", fontsize=12,
                    fontweight="bold", color="black")

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

    # Load data
    data_loader: PaperLoaderInterface = instantiate(cfg.data_loader.architecture)
    papers = data_loader.load()

    # if the key cfg.data_loader.number_of_samples is set, take n random samples
    if cfg.data_loader.number_of_samples:
        papers = random.sample(papers, cfg.data_loader.number_of_samples)

    # Handle system prompt paths if defined
    _resolve_prompt_path(cfg.material_extraction, original_cwd)
    _resolve_prompt_path(cfg.synthesis_extraction, original_cwd)

    synthesis_llms = list(cfg.synthesis_extraction.llm_names)
    judge_llms = list(cfg.judge.llm_names)

    logging.info(f"Synthesis LLMs (m={len(synthesis_llms)}): {synthesis_llms}")
    logging.info(f"Judge LLMs (n={len(judge_llms)}): {judge_llms}")

    # Build components
    # synthesis_llms drives both material + synthesis extraction (same LLM per pair)
    mat_extractors: dict[str, MaterialExtractorInterface] = {
        name: _build_component(cfg.material_extraction, name) for name in synthesis_llms
    }
    synthesis_extractors: dict[str, SynthesisExtractorInterface] = {
        name: _build_component(cfg.synthesis_extraction, name) for name in synthesis_llms
    }
    judges: dict[str, DspyGeneralSynthesisJudge] = {
        name: _build_component(cfg.judge, name) for name in judge_llms
    }

    # Result gatherer (use result_save=multi_llm config)
    result_gather = instantiate(cfg.result_save.architecture)
    result_dir = cfg.result_save.architecture.result_dir

    # LM refs for per-operation cost tracking
    mat_lms = {name: getattr(mat_extractors[name], "lm", None) for name in synthesis_llms}
    synthesis_lms = {name: getattr(synthesis_extractors[name], "lm", None) for name in synthesis_llms}
    judge_lms = {name: getattr(judges[name], "lm", None) for name in judge_llms}
    dspy_settings_lm = getattr(dspy.settings, "lm", None)

    # Papers to process (skip already-processed)
    to_process = [
        p for p in papers
        if p.id not in os.listdir(result_dir)
    ]

    if cfg.data_loader.number_of_samples:
        to_process = random.sample(to_process, cfg.data_loader.number_of_samples)

    total_cost = 0.0

    def process_paper(paper) -> tuple:
        """Process a single paper: extract, evaluate, save. Returns (summary, cost)."""
        logging.info(f"Processing {paper.name}")
        multi_llm_results = []
        eval_matrix = {}
        cost_operations = []

        initial_dspy_cost = get_lm_cost(dspy_settings_lm) if dspy_settings_lm else 0.0

        try:
            for synth_llm in synthesis_llms:
                # --- Material extraction ---
                synth_lm = synthesis_lms.get(synth_llm)
                cost_before = get_lm_cost(synth_lm) if synth_lm else 0.0

                logging.info(f"  [{synth_llm}] Material extraction")
                materials_text = mat_extractors[synth_llm].forward(
                    input=clean_text(paper.publication_text)
                )

                cost_after = get_lm_cost(synth_lm) if synth_lm else 0.0
                cost_operations.append({
                    "operation": "material_extraction",
                    "synth_llm": synth_llm,
                    "cost_usd": cost_after - cost_before,
                })

                # Filter out LLM responses that indicate no materials found
                _no_mat_phrases = {"no material", "none", "n/a", "not found", "no synthesis"}
                raw = (materials_text or "").strip()
                if any(p in raw.lower() for p in _no_mat_phrases):
                    materials = []
                else:
                    materials = [
                        m.strip()
                    for m in (materials_text or "").replace("\n", ",").split(",")
                        if m.strip()
                    ]

                if not materials:
                    logging.warning(f"No materials found for paper {paper.name} with llm {synth_llm}")
                    multi_llm_results.append({
                        "synth_llm": synth_llm,
                        "materials": [],
                        "note": "No materials found",
                    })
                    continue

                logging.info(f"  [{synth_llm}] Found materials: {materials}")

                synth_entry = {"synth_llm": synth_llm, "materials": []}
                eval_matrix[synth_llm] = {}

                for material in materials:
                    # --- Synthesis extraction ---
                    
                    cost_before = get_lm_cost(synth_lm) if synth_lm else 0.0

                    logging.info(f"  [{synth_llm}] Synthesis -> {material}")
                    try:
                        synthesis = synthesis_extractors[synth_llm].forward(
                            input=(clean_text(paper.publication_text), material)
                        )
                    
                        cost_after = get_lm_cost(synth_lm) if synth_lm else 0.0
                        cost_operations.append({
                            "operation": "synthesis_extraction",
                            "synth_llm": synth_llm,
                            "material": material,
                            "cost_usd": cost_after - cost_before,
                        })

                        # --- Evaluate with each judge LLM ---
                        evaluations = []
                        for judge_llm in judge_llms:
                            if judge_llm not in eval_matrix[synth_llm]:
                                eval_matrix[synth_llm][judge_llm] = {}

                            j_lm = judge_lms.get(judge_llm)
                            cost_before = get_lm_cost(j_lm) if j_lm else 0.0

                            logging.info(f"  Judge [{judge_llm}] on [{synth_llm}] -> {material}")
                            try:
                                evaluation = judges[judge_llm].forward((
                                    clean_text(paper.publication_text),
                                    json.dumps(synthesis.model_dump()),
                                    material,
                                ))
                                score = evaluation.scores.overall_score
                                logging.info(f"    Score: {score}/5.0")
                            except Exception as e:
                                logging.error(f"Evaluation failed for {material}: {e}")
                                evaluation = None
                                score = None

                            cost_after = get_lm_cost(j_lm) if j_lm else 0.0
                            cost_operations.append({
                                "operation": "evaluation",
                                "synth_llm": synth_llm,
                                "judge_llm": judge_llm,
                                "material": material,
                                "cost_usd": cost_after - cost_before,
                            })

                            eval_matrix[synth_llm][judge_llm][material] = score
                            evaluations.append({
                                "judge_llm": judge_llm,
                                "evaluation": evaluation.model_dump() if evaluation else None,
                                "overall_score": score,
                            })

                        

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
                        evaluations = []
                        for judge_llm in judge_llms:
                            if judge_llm not in eval_matrix[synth_llm]:
                                eval_matrix[synth_llm][judge_llm] = {}
                            eval_matrix[synth_llm][judge_llm][material] = None
                            evaluations.append({
                                "judge_llm": judge_llm,
                                "evaluation": None,
                                "overall_score": None,
                            })

                    # One entry per (synth_llm, material) with all judge evaluations
                    synth_entry["materials"].append({
                        "material": material,
                        "synthesis": synthesis.model_dump(),
                        "evaluations": evaluations,
                    })

                multi_llm_results.append(synth_entry)

            # Build summary for heatmap: avg overall_score per (synth_llm, judge_llm)
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

            # Add dspy global LM cost delta (same as single LLM example script)
            final_dspy_cost = get_lm_cost(dspy_settings_lm) if dspy_settings_lm else 0.0
            dspy_cost = (final_dspy_cost or 0.0) - (initial_dspy_cost or 0.0)
            if dspy_cost > 0:
                cost_operations.append({
                    "operation": "dspy_settings_lm",
                    "cost_usd": dspy_cost,
                })

            paper_cost = sum(op["cost_usd"] for op in cost_operations)

            #call result gather, which saves to result directory
            result_gather.gather(
                paper_id=paper.id,
                publication_text=paper.publication_text,
                si_text=paper.si_text,
                multi_llm_results=multi_llm_results,
                cost_data=cost_operations,
            )

            # Save per-paper evaluation matrix PNG
            paper_dir = os.path.join(result_dir, paper.id)
            logging.info(f"  Heatmap summary for {paper.name}: {json.dumps({s: {j: v.get('avg_overall_score') for j, v in jd.items()} for s, jd in summary.items()})}")
            _save_matrix_png(
                summary, synthesis_llms, judge_llms,
                title=f"Evaluation Matrix - {paper.name}",
                output_path=os.path.join(paper_dir, "evaluation_matrix.png"),
            )

            num_materials = sum(len(e["materials"]) for e in multi_llm_results)
            logging.info(f"Processed {num_materials} material entries across {len(multi_llm_results)} LLMs")
            logging.info(f"Paper cost: ${paper_cost:.6f}")

            return summary, paper_cost

        except Exception as e:
            logging.error(f"Failed to process paper {paper.name}: {e}")
            return None, 0.0

    all_paper_results = []
    max_workers = 4
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        logging.info(f"Processing {len(to_process)} papers")
        futures = {executor.submit(process_paper, paper): paper for paper in to_process}
        for future in as_completed(futures):
            paper = futures[future]
            try:
                summary, cost = future.result()
                if summary is not None:
                    all_paper_results.append(summary)
                    total_cost += cost
                    logging.info(f"Finished {paper.name}: cost=${cost:.6f}")
            except Exception as e:
                logging.error(f"Error processing {paper.name}: {e}")

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
            s: {j: {"avg_overall_score": round(sum(v) / len(v), 2) if v else None}
                for j, v in jd.items()}
            for s, jd in totals.items()
        }
        _save_matrix_png(
            global_summary, synthesis_llms, judge_llms,
            title=f"Evaluation Matrix (avg over {len(all_paper_results)} papers)",
            output_path=os.path.join(result_dir, "global_avg_evaluation_matrix.png"),
        )

    logging.info(f"Total cost across all papers: ${total_cost:.6f}")
    logging.info("Success")


if __name__ == "__main__":
    main()
