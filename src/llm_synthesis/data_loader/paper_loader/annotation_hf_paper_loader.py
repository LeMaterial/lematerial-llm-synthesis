from pathlib import Path

from llm_synthesis.data_loader.paper_loader.base import PaperLoaderInterface
from llm_synthesis.models.paper import Paper
from llm_synthesis.utils.paper_id_utils import (
    folder_id_to_hf_id,
    hf_id_to_folder_id,
)

DATASET_ID = "LeMaterial/LeMat-Synth-Papers"
SPLIT = "sample_for_evaluation"


class AnnotationHFLoader(PaperLoaderInterface):
    """
    Load papers from HuggingFace, restricted to IDs that have a folder under
    annotations_dir. Uses cond-mat dot/slash normalisation via paper_id_utils.
    """

    def __init__(
        self,
        annotations_dir: str = "annotations",
        dataset_uri: str = DATASET_ID,
        split: str = SPLIT,
    ):
        self.annotations_dir = Path(annotations_dir)
        self.dataset_uri = dataset_uri
        self.split = split

    def _get_folder_ids(self) -> list[str]:
        if not self.annotations_dir.is_dir():
            return []
        return sorted(
            d.name
            for d in self.annotations_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        )

    def load(self) -> list[Paper]:
        import datasets

        folder_ids = self._get_folder_ids()
        if not folder_ids:
            return []

        needed_hf = {folder_id_to_hf_id(pid) for pid in folder_ids}
        collected: dict[str, Paper] = {}

        try:
            dataset = datasets.load_dataset(
                self.dataset_uri, split=self.split, streaming=True
            )
            for row in dataset:
                hf_id = row.get("id")
                if hf_id in needed_hf:
                    folder_id = hf_id_to_folder_id(hf_id)
                    collected[folder_id] = Paper(
                        publication_text=row["text_paper"],
                        si_text=row.get("text_si") or "",
                        name=row.get("title") or folder_id,
                        id=folder_id,
                    )
                    if len(collected) == len(needed_hf):
                        break
        except (TypeError, ValueError):
            dataset = datasets.load_dataset(self.dataset_uri, split=self.split)
            for row in dataset:
                hf_id = row.get("id")
                if hf_id in needed_hf:
                    folder_id = hf_id_to_folder_id(hf_id)
                    collected[folder_id] = Paper(
                        publication_text=row["text_paper"],
                        si_text=row.get("text_si") or "",
                        name=row.get("title") or folder_id,
                        id=folder_id,
                    )

        missing = [pid for pid in folder_ids if pid not in collected]
        if missing:
            location = f"{self.dataset_uri} {self.split}"
            print(f"Missing IDs (not found in {location}): {missing}")

        return [collected[pid] for pid in folder_ids if pid in collected]
