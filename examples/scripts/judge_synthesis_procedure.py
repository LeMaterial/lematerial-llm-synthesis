import json
import logging

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from llm_synthesis.metrics.judge.base import SynthesisJudgeInterface


@hydra.main(
    config_path="../config", config_name="config.yaml", version_base=None
)
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger(__name__)

    # Instantiate the judge from the Hydra config
    judge: SynthesisJudgeInterface = instantiate(cfg.judge.architecture)
    log.info(f"Instantiated judge: {type(judge).__name__}")

    # --- Example Data ---
    target_material = "La1.93Sr0.07CuO4"

    synthesis_procedure = """ The synthesis procedure for the La1.93Sr0.07CuO<sup>4</sup> single crystal is described in the publication. The crystal was grown from 4N materials of La2O<sup>3</sup>, SrCO<sup>3</sup>, and CuO using the Traveling Solvent Floating Zone (TSFZ) technique. This technique allows for the growth of large crystals of several centimeters in length under accurately controllable stable conditions, including flux composition, temperature, and oxygen partial pressure, which are prerequisites for obtaining homogeneous crystals. 
    After the growth procedure, the sample for NMR measurements was cut from a single crystalline rod along the crystallographic a, b, and c-axis using the Laue x-ray method. The outer dimensions of the sample measured 1.7 mm, 2.5 mm, and 0.8 mm in the a, b, and c-directions, respectively."""
    
    extracted_recipe = """
   {
    "id": "La1.93Sr0.07CuO4_synthesis",
    "target_compound": "La1.93Sr0.07CuO4",
    "materials": [
        "La2O3",
        "SrCO3",
        "CuO"
    ],
    "steps": [
        {
            "action": "mix",
            "materials": [
                "La2O3",
                "SrCO3",
                "CuO"
            ],
            "conditions": {
                "temperature": null,
                "temp_unit": null,
                "duration": null,
                "time_unit": null,
                "atmosphere": null,
                "stirring": null
            }
        },
        {
            "action": "heat",
            "materials": [
                "La2O3",
                "SrCO3",
                "CuO"
            ],
            "conditions": {
                "temperature": null,
                "temp_unit": null,
                "duration": null,
                "time_unit": null,
                "atmosphere": "oxygen",
                "stirring": null
            }
        }
    ],
    "notes": "The synthesis was performed using the Traveling Solvent Floating Zone (TSFZ) technique to grow large single crystals under stable conditions. The sample for NMR measurements was cut from a single crystalline rod using the Laue x-ray method."
}
    """

    log.info("--- Evaluating Recipe ---")
    log.info(f"Target: {target_material}")
    log.info(f"Synthesis Procedure: {synthesis_procedure}")
    log.info(f"Extracted: {extracted_recipe}")
    log.info("------------------------")

    # Run the evaluation
    evaluation_input = (
        target_material,
        extracted_recipe,
        synthesis_procedure,
    )
    result = judge.forward(evaluation_input)

    # Print the results
    log.info("\n--- JUDGE'S EVALUATION ---")
    log.info("\n[Reasoning]:")
    print(result.reasoning)

    log.info("\n[Scores]:")
    print(json.dumps(result.scores.model_dump(), indent=2))

    log.info("\n--- Evaluation Complete ---")


if __name__ == "__main__":
    main() 