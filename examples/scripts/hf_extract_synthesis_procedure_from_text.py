import logging
import os
import warnings

import datasets
import hydra
from datasets import Dataset, Features, Sequence, Value
from hydra.utils import get_original_cwd, instantiate
from omegaconf import DictConfig, OmegaConf

from llm_synthesis.models.paper import (
    Paper,
    PaperWithSynthesisOntologies,
    SynthesisEntry,
)
from llm_synthesis.result_gather.synthesis_results.hf_dataset_updater import (
    HFSynthesisUpdater,
)
from llm_synthesis.transformers.material_extraction.base import (
    MaterialExtractorInterface,
)
from llm_synthesis.transformers.synthesis_extraction.base import (
    SynthesisExtractorInterface,
)
from llm_synthesis.utils import clean_text

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

logging.getLogger("pydantic").setLevel(logging.ERROR)
logging.getLogger("LiteLLM").setLevel(logging.ERROR)
logging.getLogger("litellm").setLevel(logging.ERROR)
logging.basicConfig(level=logging.INFO)


# Define the schema for the structured_synthesis column using datasets.Features
# (not sure if this is the correct way to do it @georgia ?)
SYNTHESIS_ENTRY_COL_DATASETS_FEATURES = Features(
    {
        "material": Value(dtype="string", id=None),
        "synthesis": Features(
            {  # Direct mapping of GeneralSynthesisOntology
                "target_compound": Value(dtype="string", id=None),
                "synthesis_method": Value(dtype="string", id=None),
                "starting_materials": Sequence(
                    feature=Features(
                        {
                            "vendor": Value(dtype="string", id=None),
                            "name": Value(dtype="string", id=None),
                            "amount": Value(dtype="float64", id=None),
                            "unit": Value(dtype="string", id=None),
                            "purity": Value(dtype="string", id=None),
                        }
                    )
                ),
                "steps": Sequence(
                    feature=Features(
                        {
                            "step_number": Value(dtype="int64", id=None),
                            "action": Value(dtype="string", id=None),
                            "description": Value(dtype="string", id=None),
                            "materials": Sequence(
                                feature=Features(
                                    {
                                        "vendor": Value(
                                            dtype="string", id=None
                                        ),
                                        "name": Value(dtype="string", id=None),
                                        "amount": Value(
                                            dtype="float64", id=None
                                        ),
                                        "unit": Value(dtype="string", id=None),
                                        "purity": Value(
                                            dtype="string", id=None
                                        ),
                                    }
                                )
                            ),
                            "equipment": Sequence(
                                feature=Features(
                                    {
                                        "name": Value(dtype="string", id=None),
                                        "instrument_vendor": Value(
                                            dtype="string", id=None
                                        ),
                                        "settings": Value(
                                            dtype="string", id=None
                                        ),
                                    }
                                )
                            ),
                            "conditions": Features(
                                {
                                    "temperature": Value(
                                        dtype="float64", id=None
                                    ),
                                    "temp_unit": Value(dtype="string", id=None),
                                    "duration": Value(dtype="float64", id=None),
                                    "time_unit": Value(dtype="string", id=None),
                                    "pressure": Value(dtype="float64", id=None),
                                    "pressure_unit": Value(
                                        dtype="string", id=None
                                    ),
                                    "atmosphere": Value(
                                        dtype="string", id=None
                                    ),
                                    "stirring": Value(dtype="bool", id=None),
                                    "stirring_speed": Value(
                                        dtype="float64", id=None
                                    ),
                                    "ph": Value(dtype="float64", id=None),
                                }
                            ),
                        }
                    )
                ),
                "equipment": Sequence(
                    feature=Features(
                        {
                            "name": Value(dtype="string", id=None),
                            "instrument_vendor": Value(dtype="string", id=None),
                            "settings": Value(dtype="string", id=None),
                        }
                    )
                ),
                "notes": Value(dtype="string", id=None),
            }
        ),
    }
)


@hydra.main(
    config_path="../config", config_name="config.yaml", version_base=None
)
def main(cfg: DictConfig) -> None:
    original_cwd = get_original_cwd()

    if hasattr(cfg.data_loader.architecture, "data_dir"):
        if not (
            cfg.data_loader.architecture.data_dir.startswith("s3://")
            or cfg.data_loader.architecture.data_dir.startswith("gs://")
            or cfg.data_loader.architecture.data_dir.startswith("/")
        ):
            cfg.data_loader.architecture.data_dir = os.path.join(
                original_cwd, cfg.data_loader.architecture.data_dir
            )
            OmegaConf.update(
                cfg,
                "data_loader.architecture.data_dir",
                cfg.data_loader.architecture.data_dir,
            )

    material_extractor: MaterialExtractorInterface = instantiate(
        cfg.material_extraction.architecture
    )
    synthesis_extractor: SynthesisExtractorInterface = instantiate(
        cfg.synthesis_extraction.architecture
    )

    target_split_name = (
        cfg.data_loader.architecture.split.split("[")[0]
        if "[" in cfg.data_loader.architecture.split
        else cfg.data_loader.architecture.split
    )
    dataset_uri = cfg.data_loader.architecture.dataset_uri

    skip_processed = cfg.result_save.get("skip_already_processed", False)

    logging.info(
        f"Push to hub config: {cfg.result_save.get('push_to_hub', 'NOT_FOUND')}"
    )
    logging.info(
        f"Repo ID config: {cfg.result_save.get('repo_id', 'NOT_FOUND')}"
    )
    logging.info(f"Skip processed config: {skip_processed}")
    logging.info(
        f"Loading Hugging Face Dataset split '{target_split_name}'"
        f"from '{dataset_uri}' for update."
    )

    temp_loaded_dataset = None
    try:
        temp_loaded_dataset = datasets.load_dataset(
            dataset_uri, split=target_split_name
        )
        logging.info(
            f"Loaded {len(temp_loaded_dataset)} examples for HF update."
        )
    except Exception as e:
        logging.error(
            f"Failed to load Hugging Face Dataset split '{target_split_name}'"
            f"from '{dataset_uri}': {e}"
        )
        raise

    update_column_name = "structured_synthesis"

    full_dataset_features = temp_loaded_dataset.features.copy()
    full_dataset_features["images"] = Value(dtype="string", id=None)
    full_dataset_features[update_column_name] = Sequence(
        feature=SYNTHESIS_ENTRY_COL_DATASETS_FEATURES
    )

    if update_column_name not in temp_loaded_dataset.column_names:
        logging.info(
            f"Column '{update_column_name}' not found."
            "Initializing it with empty lists and correct schema."
        )

        initial_data = temp_loaded_dataset.to_dict()
        initial_data[update_column_name] = [
            [] for _ in range(len(temp_loaded_dataset))
        ]

        # Create a NEW Dataset from scratch,
        # explicitly applying the full_dataset_features ?
        original_hf_dataset_split = Dataset.from_dict(
            initial_data, features=full_dataset_features
        )

        logging.info(
            f"Column '{update_column_name}' added with explicit features."
        )
        logging.info(
            f"Updated dataset columns: {original_hf_dataset_split.column_names}"
        )
        logging.info(
            f"Updated dataset features: {original_hf_dataset_split.features}"
        )

        if cfg.result_save.get("push_to_hub"):
            repo_id = cfg.result_save.repo_id

            logging.info(
                f"Pushing initial empty '{update_column_name}' column"
                f"to Hugging Face Hub: {repo_id}/{target_split_name}"
            )
            original_hf_dataset_split.push_to_hub(
                repo_id=repo_id,
                split=target_split_name,
            )
            logging.info(
                f"Initial empty '{update_column_name}' column"
                "pushed successfully with correct schema."
            )
        else:
            logging.warning(
                "Skipping initial push of empty column"
                "as push_to_hub is false."
            )

    else:
        logging.info(f"Column '{update_column_name}' already exists.")
        current_column_features = temp_loaded_dataset.features.get(
            update_column_name
        )
        expected_column_features = full_dataset_features.get(update_column_name)

        if current_column_features != expected_column_features:
            logging.error(
                "Despite prior steps, loaded dataset features for"
                "'structured_synthesis' still don't match expected."
            )
            logging.error(f"  Current: {current_column_features}")
            logging.error(f"  Expected: {expected_column_features}")
            logging.error(
                "This means the column's schema on Hugging Face Hub"
                "is fundamentally incompatible and was not properly"
                "removed/reset."
            )
            raise ValueError(
                "Schema mismatch persists. Please ensure"
                "'structured_synthesis' is truly deleted from Hub,"
                "and clear local cache."
            )
        else:
            logging.info(
                f"Column '{update_column_name}' exists and its schema"
                "matches the expected local schema."
            )
            original_hf_dataset_split = temp_loaded_dataset

    hf_dataset_entries_by_id = {
        str(item["id"]): item for item in original_hf_dataset_split
    }

    logging.info(
        f"Prepared {len(original_hf_dataset_split)} examples"
        "from Hub for processing."
    )

    if hasattr(
        cfg.material_extraction.architecture.lm.system_prompt, "prompt_path"
    ):
        prompt_path = os.path.join(
            original_cwd,
            cfg.material_extraction.architecture.lm.system_prompt.prompt_path,
        )
        OmegaConf.update(
            cfg,
            "material_extraction.architecture.lm.system_prompt.prompt_path",
            prompt_path,
        )

    if hasattr(
        cfg.synthesis_extraction.architecture.lm.system_prompt, "prompt_path"
    ):
        prompt_path = os.path.join(
            original_cwd,
            cfg.synthesis_extraction.architecture.lm.system_prompt.prompt_path,
        )
        OmegaConf.update(
            cfg,
            "synthesis_extraction.architecture.lm.system_prompt.prompt_path",
            prompt_path,
        )

    result_gather_for_batch = HFSynthesisUpdater(
        update_column="structured_synthesis"
    )
    processed_count_in_batch = 0

    logging.info("Starting synthesis extraction process.")

    for idx, paper_dict in enumerate(original_hf_dataset_split):
        paper = Paper(
            id=paper_dict["id"],
            name=paper_dict.get("title", f"Paper {paper_dict['id']}"),
            publication_text=paper_dict.get("text_paper"),
            si_text=paper_dict.get("text_si"),
        )

        logging.info(f"Processing {paper.name} (ID: {paper.id})")

        update_column_name = result_gather_for_batch.get_update_column_name()
        current_hf_entry = hf_dataset_entries_by_id.get(str(paper.id), {})

        if (
            skip_processed
            and current_hf_entry
            and len(current_hf_entry.get(update_column_name, [])) > 0
        ):
            logging.info(
                f"Skipping paper {paper.name} (ID: {paper.id}):"
                f"'{update_column_name}' already populated with data."
            )

            existing_synthesis_data = [
                SynthesisEntry(
                    material=s.get("material"), synthesis=s.get("synthesis")
                )
                for s in current_hf_entry.get(update_column_name, [])
            ]

            paper_with_syntheses_existing = PaperWithSynthesisOntologies(
                name=paper.name,
                id=paper.id,
                publication_text=paper.publication_text,
                si_text=paper.si_text,
                all_syntheses=existing_synthesis_data,
            )
            result_gather_for_batch.gather(paper_with_syntheses_existing)
            continue

        if not paper.publication_text:
            logging.warning(
                f"Skipping synthesis for paper {paper.id}:"
                "No 'text_paper' available."
            )
            result_gather_for_batch.gather(
                PaperWithSynthesisOntologies(
                    name=paper.name,
                    id=paper.id,
                    publication_text=None,
                    si_text=None,
                    all_syntheses=[],
                )
            )
            continue

        try:
            materials_text = material_extractor.forward(
                input=clean_text(paper.publication_text)
            )

            if materials_text:
                materials = [
                    m.strip()
                    for m in materials_text.replace("\n", ",").split(",")
                    if m.strip()
                ]
            else:
                materials = []

            logging.info(f"Found materials for paper {paper.id}: {materials}")

            all_syntheses = []
            for material in materials:
                logging.info(
                    f"Processing material: {material} for paper {paper.id}"
                )
                try:
                    structured_synthesis_procedure = (
                        synthesis_extractor.forward(
                            input=(clean_text(paper.publication_text), material)
                        )
                    )
                    synthesis_data_to_add = (
                        structured_synthesis_procedure.model_dump(
                            exclude_none=False
                        )
                        if structured_synthesis_procedure
                        else None
                    )
                    all_syntheses.append(
                        SynthesisEntry(
                            material=material, synthesis=synthesis_data_to_add
                        )
                    )
                except Exception as e:
                    logging.error(
                        f"Failed to process material {material} for paper"
                        f"{paper.name}: {e}"
                    )
                    all_syntheses.append(
                        SynthesisEntry(material=material, synthesis=None)
                    )

            paper_with_syntheses = PaperWithSynthesisOntologies(
                name=paper.name,
                id=paper.id,
                publication_text=paper.publication_text,
                si_text=paper.si_text,
                all_syntheses=all_syntheses,
            )
            result_gather_for_batch.gather(paper_with_syntheses)
            processed_count_in_batch += 1

            logging.info(
                f"Processed {len(all_syntheses)} materials for paper"
                f"{paper.name} (ID: {paper.id})"
            )
        except Exception as e:
            logging.error(
                f"Failed to process paper {paper.name} (ID: {paper.id}): {e}"
            )
            result_gather_for_batch.gather(
                PaperWithSynthesisOntologies(
                    name=paper.name,
                    id=paper.id,
                    publication_text=paper.publication_text,
                    si_text=paper.si_text,
                    all_syntheses=[],
                )
            )
            continue

        if (
            processed_count_in_batch > 0
            and processed_count_in_batch % cfg.result_save.get("batch_size", 3)
            == 0
        ):
            logging.info(
                f"[{idx + 1}/{len(original_hf_dataset_split)}] collected"
                f"data (batch of {processed_count_in_batch} processed papers)"
                "to Hugging Face Hub."
            )

            current_batch_synthesis_data = (
                result_gather_for_batch.get_accumulated_synthesis_data()
            )

            updated_examples = []
            for example in original_hf_dataset_split:
                paper_id = str(example["id"])
                updated_example = dict(example)

                if paper_id in current_batch_synthesis_data:
                    updated_example[update_column_name] = (
                        current_batch_synthesis_data[paper_id]
                    )

                if (
                    "images" in updated_example
                    and updated_example["images"] is None
                ):
                    updated_example["images"] = ""

                updated_examples.append(updated_example)

            updated_hf_dataset_to_push = datasets.Dataset.from_list(
                updated_examples, features=full_dataset_features
            )

            if cfg.result_save.get("push_to_hub"):
                repo_id = cfg.result_save.repo_id
                updated_hf_dataset_to_push.push_to_hub(
                    repo_id=repo_id,
                    split=target_split_name,
                )
                logging.info("Batch pushed to Hub successfully.")
            else:
                logging.warning("Skipping batch push as push_to_hub is false.")

            result_gather_for_batch = HFSynthesisUpdater(
                update_column="structured_synthesis"
            )
            processed_count_in_batch = 0

    if processed_count_in_batch > 0:
        logging.info(
            f"Final push of remaining {processed_count_in_batch} processed"
            "papers to Hugging Face Hub."
        )

        current_batch_synthesis_data = (
            result_gather_for_batch.get_accumulated_synthesis_data()
        )

        updated_examples = []
        for example in original_hf_dataset_split:
            paper_id = str(example["id"])
            updated_example = dict(example)

            if paper_id in current_batch_synthesis_data:
                updated_example[update_column_name] = (
                    current_batch_synthesis_data[paper_id]
                )

            if (
                "images" in updated_example
                and updated_example["images"] is None
            ):
                updated_example["images"] = ""

            updated_examples.append(updated_example)

        updated_hf_dataset_to_push = datasets.Dataset.from_list(
            updated_examples, features=full_dataset_features
        )

        if cfg.result_save.get("push_to_hub"):
            repo_id = cfg.result_save.repo_id
            updated_hf_dataset_to_push.push_to_hub(
                repo_id=repo_id,
                split=target_split_name,
            )
            logging.info("Final batch pushed to Hub successfully.")
        else:
            logging.warning("Skipping final push as push_to_hub is false.")

    logging.info("Process completed.")


if __name__ == "__main__":
    main()
