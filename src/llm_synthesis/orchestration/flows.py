"""Prefect flows for llm_synthesis pipeline orchestration."""

import json
import os
from typing import Any

import hydra.utils
from prefect import flow, get_run_logger, task

from llm_synthesis.models.paper import (
    PaperWithSynthesisOntologies,
    SynthesisEntry,
)
from llm_synthesis.orchestration.tasks import (
    evaluate_synthesis,
    extract_materials,
    extract_synthesis,
    load_papers,
    save_paper_results,
)


@task
def _process_paper(
    paper,
    material_extractor,
    synthesis_extractor,
    judge,
    result_gather,
    retries_cfg: dict[str, int],
    retry_delay: int,
) -> None:
    """Process a single paper through the full extraction pipeline."""
    logger = get_run_logger()
    logger.info(f"Processing paper: {paper.name}")

    materials = extract_materials.with_options(
        retries=retries_cfg["material_extraction"],
        retry_delay_seconds=retry_delay,
    )(material_extractor, paper.publication_text)

    if not materials:
        logger.warning(f"No materials found for paper {paper.name}")
        return

    all_syntheses = []
    for material in materials:
        synthesis = extract_synthesis.with_options(
            retries=retries_cfg["synthesis_extraction"],
            retry_delay_seconds=retry_delay,
        )(synthesis_extractor, paper.publication_text, material)
        evaluation = evaluate_synthesis.with_options(
            retries=retries_cfg["judge_evaluation"],
            retry_delay_seconds=retry_delay,
        )(
            judge,
            paper.publication_text,
            json.dumps(synthesis.model_dump()),
            material,
        )
        all_syntheses.append(
            SynthesisEntry(
                material=material,
                synthesis=synthesis,
                evaluation=evaluation,
            )
        )

    paper_with_results = PaperWithSynthesisOntologies(
        **paper.model_dump(),
        all_syntheses=all_syntheses,
    )
    save_paper_results.with_options(
        retries=retries_cfg["result_save"],
        retry_delay_seconds=retry_delay,
    )(result_gather, paper_with_results)
    logger.info(
        f"Saved {len(all_syntheses)} syntheses for paper {paper.name}"
    )


@flow
def synthesis_extraction_flow(config: dict[str, Any]) -> None:
    """Orchestrate the full synthesis extraction pipeline.

    Instantiates all pipeline components from the provided config dict,
    loads papers, and processes each one through material extraction,
    synthesis extraction, evaluation, and result saving.

    Args:
        config: Plain dict (from OmegaConf.to_container) containing all
                Hydra config sections: data_loader, material_extraction,
                synthesis_extraction, judge, result_save, orchestration.
    """
    logger = get_run_logger()

    # Instantiate components inside the flow from serializable config.
    # This avoids Prefect serialization issues with non-picklable objects
    # (dspy.LM contains httpx connection pools).
    data_loader = hydra.utils.instantiate(
        config["data_loader"]["architecture"]
    )
    material_extractor = hydra.utils.instantiate(
        config["material_extraction"]["architecture"]
    )
    synthesis_extractor = hydra.utils.instantiate(
        config["synthesis_extraction"]["architecture"]
    )
    judge = hydra.utils.instantiate(config["judge"]["architecture"])
    result_gather = hydra.utils.instantiate(
        config["result_save"]["architecture"]
    )

    orch_cfg = config.get("orchestration", {})
    retries_cfg: dict[str, int] = orch_cfg.get(
        "retries",
        {
            "material_extraction": 3,
            "synthesis_extraction": 3,
            "judge_evaluation": 2,
            "result_save": 1,
        },
    )
    retry_delay: int = orch_cfg.get("retry_delay_seconds", 5)

    result_dir = config["result_save"]["architecture"].get("result_dir", "")
    already_processed: set[str] = set()
    if result_dir and os.path.isdir(str(result_dir)):
        already_processed = set(os.listdir(str(result_dir)))

    papers = load_papers(data_loader)

    to_process = [p for p in papers if p.id not in already_processed]
    logger.info(
        f"Processing {len(to_process)} papers "
        f"({len(papers) - len(to_process)} already done)"
    )

    futures = [
        _process_paper.submit(
            paper,
            material_extractor,
            synthesis_extractor,
            judge,
            result_gather,
            retries_cfg,
            retry_delay,
        )
        for paper in to_process
    ]
    for f in futures:
        f.result()
