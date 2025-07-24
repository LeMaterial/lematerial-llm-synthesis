import fsspec

from llm_synthesis.data_loader.paper_loader.base import PaperLoaderInterface
from llm_synthesis.models.paper import Paper


class FSPaperLoader(PaperLoaderInterface):
    """
    Paper loader that loads papers from a file system.
    """

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.fs, _, _ = fsspec.get_fs_token_paths(data_dir)

    def load(self) -> list[Paper]:
        """
        Load papers from the file system.

        Returns:
            list[Paper]: A list of papers.
        """
        papers = []
        for file in self.fs.ls(self.data_dir):
            if file.endswith("SI.txt"):
                continue

            with self.fs.open(file, "r", encoding="utf-8", errors="replace") as f:
                publication_text = f.read()

            si_file = file.replace(".txt", "_SI.txt")
            if self.fs.exists(si_file):
                with self.fs.open(si_file, "r", encoding="utf-8", errors="replace") as f:
                    si_text = f.read()
            else:
                si_text = ""

            paper = Paper(
                publication_text=publication_text,
                si_text=si_text,
                name=file.split("/")[-1].split(".")[0],
                id=file.split("/")[-1].split(".")[0],
            )
            papers.append(paper)
        return papers
