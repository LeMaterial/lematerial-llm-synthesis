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
    judge: DspyGeneralSynthesisJudge = instantiate(
        cfg.judge.architecture
    )
    result_gather: ResultGatherInterface[PaperWithSynthesisOntologies] = (
        instantiate(cfg.result_save.architecture)
    )

    # Initialize cost tracking
    print("=" * 50)
    print("STARTING LLM COST TRACKING")
    print("=" * 50)
    
    # Get LMs from all components to track costs
    synthesis_lm = getattr(synthesis_extractor, 'lm', None)
    material_lm = getattr(material_extractor, 'lm', None)
    judge_lm = getattr(judge, 'lm', None)
    
    # Also check DSPy global settings
    dspy_settings_lm = getattr(dspy.settings, 'lm', None)
    
    print(f"Synthesis LM: {synthesis_lm}")
    print(f"Material LM: {material_lm}")
    print(f"Judge LM: {judge_lm}")
    print(f"DSPy settings LM: {dspy_settings_lm}")
    
    # Track initial costs - try multiple approaches
    initial_synthesis_cost = get_lm_cost(synthesis_lm) if synthesis_lm else 0.0
    initial_material_cost = get_lm_cost(material_lm) if material_lm else 0.0
    initial_judge_cost = get_lm_cost(judge_lm) if judge_lm else 0.0
    initial_dspy_cost = get_lm_cost(dspy_settings_lm) if dspy_settings_lm else 0.0
    
    print(f"Initial synthesis LM cost: ${initial_synthesis_cost or 0.0:.6f}")
    print(f"Initial material LM cost: ${initial_material_cost or 0.0:.6f}")
    print(f"Initial judge LM cost: ${initial_judge_cost or 0.0:.6f}")
    print(f"Initial DSPy settings LM cost: ${initial_dspy_cost or 0.0:.6f}")

    # Process each paper
    for paper in papers:
        logging.info(f"Processing {paper.name}")

        try:
            # Extract list of synthesized materials
            materials_text = material_extractor.forward(
                input=clean_text(paper.publication_text)
            )

            # Parse the materials text into a list
            if materials_text:
                materials = [
                    material.strip()
                    for material in materials_text.replace("\n", ",").split(
                        ","
                    )
                    if material.strip()
                ]
            else:
                materials = []

            logging.info(f"Found materials: {materials}")

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

                    logging.info(
                        f"Extracted synthesis ontology for {material}"
                    )
                    logging.info(structured_synthesis_procedure)

                    # Evaluate the extracted synthesis procedure
                    try:
                        evaluation_input = (
                            clean_text(paper.publication_text),
                            json.dumps(structured_synthesis_procedure.model_dump()),
                            material,
                        )
                        evaluation = judge.forward(evaluation_input)
                        logging.info(
                            f"  Evaluation score: {evaluation.scores.overall_score}/5.0"
                        )
                    except Exception as e:
                        logging.error(f"Failed to evaluate synthesis for {material}: {e}")
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
                    logging.error(
                        f"Failed to process material {material}: {e}"
                    )
                    # Add a failed synthesis entry
                    all_syntheses.append(SynthesisEntry(material=material, synthesis=None, evaluation=None))

            # Calculate costs for this paper
            final_synthesis_cost_paper = get_lm_cost(synthesis_lm) if synthesis_lm else 0.0
            final_material_cost_paper = get_lm_cost(material_lm) if material_lm else 0.0
            final_judge_cost_paper = get_lm_cost(judge_lm) if judge_lm else 0.0
            final_dspy_cost_paper = get_lm_cost(dspy_settings_lm) if dspy_settings_lm else 0.0

            paper_synthesis_cost = (final_synthesis_cost_paper or 0.0) - (initial_synthesis_cost or 0.0)
            paper_material_cost = (final_material_cost_paper or 0.0) - (initial_material_cost or 0.0)
            paper_judge_cost = (final_judge_cost_paper or 0.0) - (initial_judge_cost or 0.0)
            paper_dspy_cost = (final_dspy_cost_paper or 0.0) - (initial_dspy_cost or 0.0)
            paper_total_cost = paper_synthesis_cost + paper_material_cost + paper_judge_cost + paper_dspy_cost

            # Count LLM calls for this paper
            synthesis_calls = len([s for s in all_syntheses if s.synthesis is not None])
            judge_calls = len([s for s in all_syntheses if s.evaluation is not None])

            # Prepare cost data for this paper
            cost_data = {
                "total_cost": paper_total_cost,
                "breakdown": {
                    "synthesis_extraction": paper_synthesis_cost,
                    "material_extraction": paper_material_cost,
                    "judge_evaluation": paper_judge_cost,
                    "dspy_settings": paper_dspy_cost
                },
                "summary": f"Total cost: ${paper_total_cost:.6f} for processing {len(materials)} materials",
                "models": {
                    "synthesis_extractor": getattr(synthesis_lm, 'model', 'Unknown') if synthesis_lm else 'None',
                    "material_extractor": getattr(material_lm, 'model', 'Unknown') if material_lm else 'None',
                    "judge": getattr(judge_lm, 'model', 'Unknown') if judge_lm else 'None'
                },
                "total_calls": synthesis_calls + judge_calls + 1,  # +1 for material extraction
                "materials_count": len(materials),
                "synthesis_calls": synthesis_calls,
                "material_calls": 1,
                "judge_calls": judge_calls
            }

            # Create paper object with all syntheses
            paper_with_syntheses = PaperWithSynthesisOntologies(
                name=paper.name,
                id=paper.id,
                publication_text=paper.publication_text,
                si_text=paper.si_text,
                all_syntheses=all_syntheses,
            )

            # Save results with cost data
            result_gather.gather(paper_with_syntheses, cost_data)

            logging.info(
                f"Processed {len(all_syntheses)} materials: "
                f"{[s.material for s in all_syntheses]}"
            )

        except Exception as e:
            logging.error(f"Failed to process paper {paper.name}: {e}")
            continue

    # Report final costs
    print("=" * 50)
    print("FINAL LLM COST REPORT")
    print("=" * 50)
    
    # Calculate final costs for each component
    final_synthesis_cost = get_lm_cost(synthesis_lm) if synthesis_lm else 0.0
    final_material_cost = get_lm_cost(material_lm) if material_lm else 0.0
    final_judge_cost = get_lm_cost(judge_lm) if judge_lm else 0.0
    final_dspy_cost = get_lm_cost(dspy_settings_lm) if dspy_settings_lm else 0.0
    
    # Calculate session costs
    synthesis_session_cost = (final_synthesis_cost or 0.0) - (initial_synthesis_cost or 0.0)
    material_session_cost = (final_material_cost or 0.0) - (initial_material_cost or 0.0)
    judge_session_cost = (final_judge_cost or 0.0) - (initial_judge_cost or 0.0)
    dspy_session_cost = (final_dspy_cost or 0.0) - (initial_dspy_cost or 0.0)
    total_session_cost = synthesis_session_cost + material_session_cost + judge_session_cost + dspy_session_cost
    
    print(f"Synthesis extractor session cost: ${synthesis_session_cost:.6f}")
    print(f"Material extractor session cost: ${material_session_cost:.6f}")
    print(f"Judge session cost: ${judge_session_cost:.6f}")
    print(f"DSPy settings session cost: ${dspy_session_cost:.6f}")
    print(f"Total session cost: ${total_session_cost:.6f}")
    
    if total_session_cost > 0:
        print(f"\n💰 Total cost for this run: ${total_session_cost:.6f}")
        print(f"   - Synthesis extraction: ${synthesis_session_cost:.6f}")
        print(f"   - Material extraction: ${material_session_cost:.6f}")
        print(f"   - Quality judging: ${judge_session_cost:.6f}")
        print(f"   - DSPy settings: ${dspy_session_cost:.6f}")
    else:
        print(f"\n💰 No cost data available or no LLM calls made")
        print("   This might be because:")
        print("   1. The LLM provider doesn't return cost information")
        print("   2. Cost tracking is not properly configured")
        print("   3. Calls were made but cost extraction failed")
    
    print("=" * 50)

    logging.info("Success")


if __name__ == "__main__":
    main()
