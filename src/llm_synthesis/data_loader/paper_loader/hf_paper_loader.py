import datasets

from llm_synthesis.data_loader.paper_loader.base import PaperLoaderInterface
from llm_synthesis.models.paper import Paper


class HFLoader(PaperLoaderInterface):
    """
    Paper loader that loads papers from a Hugging Face Dataset.
    """

    def __init__(self, dataset_uri: str, split: str = "train"):
        self.dataset_uri = dataset_uri
        self.split = split

    def load(self) -> list[Paper]:
        """
        Load papers from the Hugging Face Dataset.
        """
        dataset = datasets.load_dataset(self.dataset_uri, split=self.split)
        papers = []
        for paper in dataset:
            papers.append(
                Paper(
                    publication_text=paper["markdown_text"],
                    si_text="",
                    name=paper["title"],
                    id=paper["id"],
                )
            )
        return papers
