"""
Simple test script for the GeneralSynthesisOntology judge.
"""

import json
import logging

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from llm_synthesis.metrics.judge.general_synthesis_judge import (
    DspyGeneralSynthesisJudge,
)


@hydra.main(
    config_path="../config", config_name="config.yaml", version_base=None
)
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger(__name__)

    # Instantiate the judge
    judge: DspyGeneralSynthesisJudge = instantiate(cfg.judge.architecture)
    log.info(f"Instantiated judge: {type(judge).__name__}")

    # Simple test case
    source_text = """
    To synthesize lithium iron phosphate, 2.0 g of LiOH was mixed with 3.0 g 
    of FeSO4 and 2.5 g of NH4H2PO4 in 100 mL of distilled water. The mixture 
    was stirred for 30 minutes at room temperature using a magnetic stirrer. 
    The solution was then transferred to a 150 mL autoclave and heated at 
    180°C for 10 hours. After cooling, the product was filtered and dried 
    at 80°C for 6 hours.
    """

    extracted_ontology = {
        "target_compound": "lithium iron phosphate",
        "synthesis_method": "hydrothermal",
        "starting_materials": [
            {"name": "LiOH", "amount": 2.0, "unit": "g"},
            {"name": "FeSO4", "amount": 3.0, "unit": "g"},
            {"name": "NH4H2PO4", "amount": 2.5, "unit": "g"},
            {"name": "distilled water", "amount": 100, "unit": "mL"},
        ],
        "steps": [
            {
                "step_number": 1,
                "action": "mix",
                "description": "Mix precursors in water",
                "materials": [
                    {"name": "LiOH", "amount": 2.0, "unit": "g"},
                    {"name": "FeSO4", "amount": 3.0, "unit": "g"},
                    {"name": "NH4H2PO4", "amount": 2.5, "unit": "g"},
                    {"name": "distilled water", "amount": 100, "unit": "mL"},
                ],
                "equipment": [{"name": "magnetic stirrer"}],
                "conditions": {
                    "duration": 30,
                    "time_unit": "min",
                    "temperature": 25,
                    "temp_unit": "C",
                    "stirring": True,
                },
            },
            {
                "step_number": 2,
                "action": "heat",
                "description": "Hydrothermal treatment",
                "equipment": [{"name": "autoclave", "settings": "150 mL"}],
                "conditions": {
                    "temperature": 180,
                    "temp_unit": "C",
                    "duration": 10,
                    "time_unit": "h",
                },
            },
            {
                "step_number": 3,
                "action": "filter",
                "description": "Filter the product",
            },
            {
                "step_number": 4,
                "action": "dry",
                "description": "Dry the product",
                "conditions": {
                    "temperature": 80,
                    "temp_unit": "C",
                    "duration": 6,
                    "time_unit": "h",
                },
            },
        ],
        "equipment": [
            {"name": "magnetic stirrer"},
            {"name": "autoclave", "settings": "150 mL"},
        ],
    }

    extracted_ontology_json = json.dumps(extracted_ontology, indent=2)
    target_material = "lithium iron phosphate"

    log.info("--- Testing General Synthesis Judge ---")
    log.info(f"Source text length: {len(source_text)} characters")
    log.info(f"Target material: {target_material}")
    log.info("----------------------------------------")

    try:
        # Perform evaluation
        evaluation_input = (
            source_text,
            extracted_ontology_json,
            target_material,
        )
        result = judge.forward(evaluation_input)

        # Print results
        log.info("\n--- EVALUATION RESULTS ---")
        log.info(f"Overall Score: {result.scores.overall_score}/5.0")
        log.info(f"Confidence Level: {result.confidence_level}")

        log.info("\n[Individual Scores]:")
        scores = result.scores
        print(
            f"  Structural Completeness: "
            f"{scores.structural_completeness_score}/5.0"
        )
        print(f"  Material Extraction: {scores.material_extraction_score}/5.0")
        print(f"  Process Steps: {scores.process_steps_score}/5.0")
        print(
            f"  Equipment Extraction: "
            f"{scores.equipment_extraction_score}/5.0"
        )
        print(
            f"  Conditions Extraction: "
            f"{scores.conditions_extraction_score}/5.0"
        )
        print(f"  Semantic Accuracy: {scores.semantic_accuracy_score}/5.0")
        print(f"  Format Compliance: {scores.format_compliance_score}/5.0")

        if result.missing_information:
            log.info(
                f"\n[Missing Information] ({len(result.missing_information)}):"
            )
            for item in result.missing_information:
                print(f"  - {item}")

        if result.extraction_errors:
            log.info(
                f"\n[Extraction Errors] ({len(result.extraction_errors)}):"
            )
            for error in result.extraction_errors:
                print(f"  - {error}")

        if result.improvement_suggestions:
            log.info(
                f"\n[Improvement Suggestions] "
                f"({len(result.improvement_suggestions)}):"
            )
            for suggestion in result.improvement_suggestions:
                print(f"  - {suggestion}")

        log.info("\n[High-level Reasoning]:")
        print(result.reasoning)

        # Save result
        with open("test_evaluation_result.json", "w") as f:
            json.dump(result.model_dump(), f, indent=2)

        log.info("\nResult saved to: test_evaluation_result.json")

    except Exception as e:
        log.error(f"Error during evaluation: {e}")
        raise

    log.info("\n--- Test Complete ---")


if __name__ == "__main__":
    main()
