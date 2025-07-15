import logging
import os
import warnings

import hydra
from hydra.utils import get_original_cwd, instantiate
from omegaconf import DictConfig

from llm_synthesis.data_loader.paper_loader.base import PaperLoaderInterface
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
    material_specific_synthesis_extractor: SynthesisExtractorInterface = (
        instantiate(cfg.synthesis_extraction.architecture)
    )
    result_gather: ResultGatherInterface[PaperWithSynthesisOntologies] = (
        instantiate(cfg.result_save.architecture)
    )

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
                        material_specific_synthesis_extractor.forward(
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

                    # Store material and its synthesis
                    all_syntheses.append(
                        SynthesisEntry(
                            material=material,
                            synthesis=structured_synthesis_procedure,
                        )
                    )
                except Exception as e:
                    logging.error(
                        f"Failed to process material {material}: {e}"
                    )
                    # Add a failed synthesis entry
                    all_syntheses.append(
                        SynthesisEntry(material=material, synthesis=None)
                    )

            # Create paper object with all syntheses
            paper_with_syntheses = PaperWithSynthesisOntologies(
                name=paper.name,
                id=paper.id,
                publication_text=paper.publication_text,
                si_text=paper.si_text,
                all_syntheses=all_syntheses,
            )

            result_gather.gather(paper_with_syntheses)

            logging.info(
                f"Processed {len(all_syntheses)} materials: "
                f"{[s.material for s in all_syntheses]}"
            )

        except Exception as e:
            logging.error(f"Failed to process paper {paper.name}: {e}")
            continue

    logging.info("Success")


if __name__ == "__main__":
    main()
