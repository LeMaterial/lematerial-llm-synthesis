import hydra
import logging
from hydra.utils import instantiate
import json
from typing import Tuple
import dspy
from omegaconf import DictConfig
from llm_synthesis.metrics import DummyMetric
from llm_synthesis.utils import configure_dspy
import os


def load_data(cfg: DictConfig):
    # TODO: for now, only local data is implemented
    if cfg.data.location != "local":
        raise NotImplementedError("Only local data is implemented for now")
    files = os.listdir(cfg.data.directory)
    data = {}
    for file in files:
        stem_name = file.split(".")[0]
        if stem_name.endswith("_SI"):
            continue
        data[stem_name] = {
            "publication_text": open(
                os.path.join(cfg.data.directory, file), "r"
            ).read(),
            "si_text": open(
                os.path.join(cfg.data.directory, file.replace(".txt", "_SI.txt")), "r"
            ).read(),
        }

    return data


def load_models(cfg: DictConfig) -> Tuple[dspy.Module, dspy.Module]:
    configure_dspy(
        lm=cfg.synthesis_extraction.lm.name,
        model_kwargs={"temperature": cfg.synthesis_extraction.lm.temperature},
    )
    # TODO: load different LLMs for paragraph extraction and synthesis extraction
    if cfg.data.preprocessing != "paragraph_extraction":
        raise NotImplementedError("Only paragraph extraction is implemented for now")
    paragraph_parser = instantiate(cfg.paragraph_extraction.architecture)
    synthesis_parser = instantiate(cfg.synthesis_extraction.architecture)
    return paragraph_parser, synthesis_parser


@hydra.main(config_path="../config", config_name="config.yaml", version_base=None)
def main(cfg: DictConfig) -> None:
    # Load dataset for evaluation (pubs, sis, ground truth paragraphs)
    data = load_data(cfg)

    # Load model (= LLM + architecture)
    paragraph_parser, synthesis_parser = load_models(cfg)
    metric = DummyMetric()

    for key, value in data.items():
        logging.info(f"Processing {key}")
        synthesis_paragraphs = paragraph_parser(
            publication_text=value["publication_text"], si_text=value["si_text"]
        )["synthesis_paragraphs"]
        logging.info(synthesis_paragraphs)
        structured_synthesis_procedure = synthesis_parser(
            synthesis_procedure=synthesis_paragraphs
        )["structured_synthesis_procedure"]
        logging.info(structured_synthesis_procedure)
        performance = metric(
            structured_synthesis_procedure, ""
        )  # TODO: this metric is random for now
        logging.info(f"Metric: {performance}")

        # Save synthesis paragraph to .txt file
        with open(os.path.join("synthesis_paragraph.txt"), "w") as f:
            f.write(synthesis_paragraphs)

        # Save structured synthesis procedure to .json file
        with open(os.path.join("structured_synthesis_procedure.json"), "w") as f:
            f.write(json.dumps(structured_synthesis_procedure.model_dump(), indent=4))

    # if script runs correctly, log success
    logging.info("Success")


if __name__ == "__main__":
    main()
