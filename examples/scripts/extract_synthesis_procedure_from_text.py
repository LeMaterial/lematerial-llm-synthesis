import logging
import os

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from llm_synthesis.data_loader.paper_loader.base import PaperLoaderInterface
from llm_synthesis.models.paper import PaperWithSynthesisOntology
from llm_synthesis.transformers.synthesis_extraction.base import (
    StructuredSynthesisExtractorInterface,
)
from llm_synthesis.transformers.text_extraction.base import (
    TextExtractorInterface,
)


@hydra.main(
    config_path="../config", config_name="config.yaml", version_base=None
)
def main(cfg: DictConfig) -> None:
    # Load dataset for evaluation (pubs, sis, ground truth paragraphs)
    data_loader: PaperLoaderInterface = instantiate(
        cfg.data_loader.architecture
    )
    papers = data_loader.load()

    paragraph_extractor: TextExtractorInterface = instantiate(
        cfg.paragraph_extraction.architecture
    )
    synthesis_extractor: StructuredSynthesisExtractorInterface = instantiate(
        cfg.synthesis_extraction.architecture
    )

    for paper in papers:
        logging.info(f"Processing {paper.name}")
        synthesis_paragraph = paragraph_extractor.extract(
            input=paper.publication_text,
        )

        os.makedirs(paper.id, exist_ok=True)

        structured_synthesis_procedure = synthesis_extractor.extract(
            input=synthesis_paragraph,
        )
        logging.info(structured_synthesis_procedure)
        paper_enriched = PaperWithSynthesisOntology(
            **paper.model_dump(),
            synthesis_paragraph=synthesis_paragraph,
            synthesis_ontology=structured_synthesis_procedure,
        )

        os.makedirs(paper.id, exist_ok=True)

        with open(os.path.join(paper.id, "result.json"), "w") as f:
            f.write(paper_enriched.model_dump_json())

    logging.info("Success")


if __name__ == "__main__":
    main()
