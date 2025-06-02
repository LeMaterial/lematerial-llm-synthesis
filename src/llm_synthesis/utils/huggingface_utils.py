from typing import Dict
from datasets import load_dataset
from omegaconf import DictConfig


def load_huggingface_data(cfg: DictConfig) -> Dict[str, Dict[str, str]]:
    """
    Loads data from Hugging Face.

    Args:
        cfg: Configuration object containing dataset information

    Returns:
        Dict[str, Dict[str, str]]: Dictionary with id as key and markdown_text as value
    """
    dataset = load_dataset(cfg.data_fetching.dataset_name, split=cfg.data_fetching.split)

    # save the id, markdown_text as key-value pairs.
    data = {item["id"]: {"publication_text": item["markdown_text"]} for item in dataset}

    return data
