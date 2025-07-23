import json
import logging
import os
import warnings

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
from llm_synthesis.services.pipelines.process_pdf_folder_pipeline import (
    ProcessPDFFolderPipeline,
)
from llm_synthesis.services.storage.file_storage_factory import (
    create_file_storage,
)
from llm_synthesis.transformers.material_extraction.base import (
    MaterialExtractorInterface,
)
from llm_synthesis.transformers.pdf_extraction.pdf_extractor_factory import (
    PDFExtractorEnum,
    create_pdf_extractor,
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


def extract_text_from_pdfs(
    pdf_dir: str, txt_dir: str, extraction_process: str = "docling"
) -> None:
    """Extract text from PDFs in given directory and save to txt."""
    logging.info(f"Extracting text from PDFs in {pdf_dir} to {txt_dir}")

    file_storage = create_file_storage(pdf_dir)
    file_storage.create_dir(txt_dir)

    # Create PDF extractor
    pdf_extractor_enum = PDFExtractorEnum(extraction_process)
    pdf_extractor = create_pdf_extractor(pdf_extractor_enum)

    # Run the pipeline
    pipeline = ProcessPDFFolderPipeline(
        file_storage=file_storage,
        pdf_extractor=pdf_extractor,
        input_dir=pdf_dir,
        output_dir=txt_dir,
    )

    pipeline.run()
    logging.info(f"Text extraction completed. Files saved to {txt_dir}")


@hydra.main(
    config_path="../config", config_name="config.yaml", version_base=None
)
def main(cfg: DictConfig) -> None:
    original_cwd = get_original_cwd()

    # Check if we need to extract text from PDFs first
    pdf_dir = getattr(cfg.text_extraction, "pdf_dir", None)
    txt_dir = getattr(cfg.text_extraction, "txt_dir", None)
    extraction_process = getattr(
        cfg.text_extraction, "extraction_process", "docling"
    )

    if pdf_dir and txt_dir:
        # Convert relative paths to absolute if needed
        if not pdf_dir.startswith("/"):
            pdf_dir = os.path.join(original_cwd, pdf_dir)
        if not txt_dir.startswith("/"):
            txt_dir = os.path.join(original_cwd, txt_dir)

        # Extract text from PDFs first
        extract_text_from_pdfs(pdf_dir, txt_dir, extraction_process)

        # Update the data_loader config to use the extracted text directory
        if hasattr(cfg.data_loader.architecture, "data_dir"):
            cfg.data_loader.architecture.data_dir = txt_dir

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
                    for material in materials_text.replace("\n", ",").split(",")
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
                    # Add a failed synthesis entry
                    all_syntheses.append(
                        SynthesisEntry(
                            material=material, synthesis=None, evaluation=None
                        )
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
