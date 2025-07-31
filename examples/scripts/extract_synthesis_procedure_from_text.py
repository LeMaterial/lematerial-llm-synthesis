import json
import logging
import os
import warnings

import dspy
import hydra
from hydra.utils import get_original_cwd, instantiate
from omegaconf import DictConfig

from llm_synthesis.data_loader.paper_loader.base import PaperLoaderInterface
from llm_synthesis.metrics.judge.general_synthesis_judge import (
    DspyGeneralSynthesisJudge,
)
from llm_synthesis.models.ontologies.general import GeneralSynthesisOntology
from llm_synthesis.models.paper import (
    PaperWithSynthesisOntologies,
    SynthesisEntry,
)
from llm_synthesis.result_gather.base import ResultGatherInterface
from llm_synthesis.transformers.material_extraction.base import (
    MaterialExtractorInterface,
)
from llm_synthesis.transformers.synthesis_extraction.base import (
    SynthesisExtractorInterface,
)
from llm_synthesis.utils import clean_text
from llm_synthesis.utils.dspy_utils import get_lm_cost

# Disable Pydantic warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

# Configure logging to reduce noise
logging.getLogger("pydantic").setLevel(logging.ERROR)
logging.getLogger("LiteLLM").setLevel(logging.ERROR)
logging.getLogger("litellm").setLevel(logging.ERROR)


@hydra.main(
    config_path="../config", config_name="config.yaml", version_base=None
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
    data_loader: PaperLoaderInterface = instantiate(
        cfg.data_loader.architecture
    )
    papers = data_loader.load()

    # Handle system prompt path if defined
    if hasattr(
        cfg.material_extraction.architecture.lm.system_prompt, "prompt_path"
    ):
        prompt_path = os.path.join(
            original_cwd,
            cfg.material_extraction.architecture.lm.system_prompt.prompt_path,
        )
        cfg.material_extraction.architecture.lm.system_prompt.prompt_path = (
            prompt_path
        )

    if hasattr(
        cfg.synthesis_extraction.architecture.lm.system_prompt, "prompt_path"
    ):
        prompt_path = os.path.join(
            original_cwd,
            cfg.synthesis_extraction.architecture.lm.system_prompt.prompt_path,
        )
        cfg.synthesis_extraction.architecture.lm.system_prompt.prompt_path = (
            prompt_path
        )

    # Initialize material extractor and material-specific synthesis extractor
    material_extractor: MaterialExtractorInterface = instantiate(
        cfg.material_extraction.architecture
    )
    synthesis_extractor: SynthesisExtractorInterface = instantiate(
        cfg.synthesis_extraction.architecture
    )
    judge: DspyGeneralSynthesisJudge = instantiate(cfg.judge.architecture)
    result_gather: ResultGatherInterface[PaperWithSynthesisOntologies] = (
        instantiate(cfg.result_save.architecture)
    )

    # Get LMs from all components to track costs
    synthesis_lm = getattr(synthesis_extractor, "lm", None)
    material_lm = getattr(material_extractor, "lm", None)
    judge_lm = getattr(judge, "lm", None)

    # Also check DSPy global settings
    dspy_settings_lm = getattr(dspy.settings, "lm", None)

    # Track initial costs - try multiple approaches
    initial_synthesis_cost = get_lm_cost(synthesis_lm) if synthesis_lm else 0.0
    initial_material_cost = get_lm_cost(material_lm) if material_lm else 0.0
    initial_judge_cost = get_lm_cost(judge_lm) if judge_lm else 0.0
    initial_dspy_cost = (
        get_lm_cost(dspy_settings_lm) if dspy_settings_lm else 0.0
    )

    # Process each paper
    total_cost = 0.0
    for paper in papers:
        logging.info(f"Processing {paper.name}")
        
        if paper.id in os.listdir(cfg.result_save.architecture.result_dir):
            logging.info(f"Skipping {paper.id} because already processed.")
            continue

        try:
            # Extract list of synthesized materials
            materials_text = material_extractor.forward(
                input=clean_text(paper.publication_text)
            )

            # Parse the materials text into a list
            if materials_text:
                materials = [
                    material.strip()
                    for material in materials_text.replace("\n", ",").split(",")
                    if material.strip()
                ]
            else:
                materials = []

            logging.info(f"Found materials: {materials}")

            # Skip processing if no materials found
            if not materials:
                logging.warning(f"No materials found for paper {paper.name}")
                continue

            # Process each material and collect all syntheses
            all_syntheses = []
            for material in materials:
                logging.info(f"Processing material: {material}")

                try:
                    # Extract synthesis procedure for specific material
                    # Pass the entire paper text + material name
                    structured_synthesis_procedure = (
                        synthesis_extractor.forward(
                            input=(
                                clean_text(paper.publication_text),
                                material,
                            ),
                        )
                    )

                    logging.info(f"Extracted synthesis ontology for {material}")
                    logging.info(structured_synthesis_procedure)

                    # Evaluate the extracted synthesis procedure
                    try:
                        evaluation_input = (
                            clean_text(paper.publication_text),
                            json.dumps(
                                structured_synthesis_procedure.model_dump()
                            ),
                            material,
                        )
                        evaluation = judge.forward(evaluation_input)
                        logging.info(
                            f"  Eval sc: {evaluation.scores.overall_score}/5.0"
                        )
                    except Exception as e:
                        logging.error(
                            f"Failed to evaluate synthesis for {material}: {e}"
                        )
                        evaluation = None

                    # Store material and its synthesis
                    all_syntheses.append(
                        SynthesisEntry(
                            material=material,
                            synthesis=structured_synthesis_procedure,
                            evaluation=evaluation,
                        )
                    )
                except Exception as e:
                    logging.error(f"Failed to process material {material}: {e}")
                    # Create a minimal synthesis entry with error information
                    failed_synthesis = GeneralSynthesisOntology(
                        target_compound=material,
                        target_compound_type="other",
                        synthesis_method="other",
                        starting_materials=[],
                        steps=[],
                        equipment=[],
                        notes=f"Processing failed: {e!s}",
                    )
                    all_syntheses.append(
                        SynthesisEntry(
                            material=material,
                            synthesis=failed_synthesis,
                            evaluation=None,
                        )
                    )

            # Calculate costs for this paper
            final_synthesis_cost_paper = (
                get_lm_cost(synthesis_lm) if synthesis_lm else 0.0
            )
            final_material_cost_paper = (
                get_lm_cost(material_lm) if material_lm else 0.0
            )
            final_judge_cost_paper = get_lm_cost(judge_lm) if judge_lm else 0.0
            final_dspy_cost_paper = (
                get_lm_cost(dspy_settings_lm) if dspy_settings_lm else 0.0
            )

            paper_synthesis_cost = (final_synthesis_cost_paper or 0.0) - (
                initial_synthesis_cost or 0.0
            )
            paper_material_cost = (final_material_cost_paper or 0.0) - (
                initial_material_cost or 0.0
            )
            paper_judge_cost = (final_judge_cost_paper or 0.0) - (
                initial_judge_cost or 0.0
            )
            paper_dspy_cost = (final_dspy_cost_paper or 0.0) - (
                initial_dspy_cost or 0.0
            )
            paper_total_cost = (
                paper_synthesis_cost
                + paper_material_cost
                + paper_judge_cost
                + paper_dspy_cost
            )

            # Count LLM calls for this paper
            synthesis_calls = len(
                [s for s in all_syntheses if s.synthesis is not None]
            )
            judge_calls = len(
                [s for s in all_syntheses if s.evaluation is not None]
            )

            # Prepare cost data for this paper
            cost_data = {
                "total_cost": paper_total_cost,
                "breakdown": {
                    "synthesis_extraction": paper_synthesis_cost,
                    "material_extraction": paper_material_cost,
                    "judge_evaluation": paper_judge_cost,
                    "dspy_settings": paper_dspy_cost,
                },
                "models": {
                    "synthesis_extractor": getattr(
                        synthesis_lm, "model", "Unknown"
                    )
                    if synthesis_lm
                    else "None",
                    "material_extractor": getattr(
                        material_lm, "model", "Unknown"
                    )
                    if material_lm
                    else "None",
                    "judge": getattr(judge_lm, "model", "Unknown")
                    if judge_lm
                    else "None",
                },
                "total_calls": synthesis_calls
                + judge_calls
                + 1,  # +1 for material extraction
                "materials_count": len(materials),
                "synthesis_calls": synthesis_calls,
                "material_calls": 1,
                "judge_calls": judge_calls,
            }

            # Create paper object with all syntheses
            paper_with_syntheses = PaperWithSynthesisOntologies(
                name=paper.name,
                id=paper.id,
                publication_text=paper.publication_text,
                si_text=paper.si_text,
                all_syntheses=all_syntheses,
                cost_data=cost_data,
            )

            # Save results with cost data
            result_gather.gather(paper_with_syntheses)

            # Add this paper's cost to total
            total_cost += paper_total_cost

            logging.info(
                f"Processed {len(all_syntheses)} materials: "
                f"{[s.material for s in all_syntheses]}"
            )
            logging.info(f"Paper cost: ${paper_total_cost:.6f}")

        except Exception as e:
            logging.error(f"Failed to process paper {paper.name}: {e}")
            continue

    # Print final total cost
    logging.info(f"Total cost across all papers: ${total_cost:.6f}")
    logging.info("Success")


if __name__ == "__main__":
    main()
