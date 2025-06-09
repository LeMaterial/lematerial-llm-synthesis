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
            paper = Paper(
                publication_text=self.fs.open(file, "r").read(),
                si_text=self.fs.open(
                    file.replace(".txt", "_SI.txt"), "r"
                ).read()
                if self.fs.exists(file.replace(".txt", "_SI.txt"))
                else "",
                name=file.split("/")[-1].split(".")[0],
                id=file.split("/")[-1].split(".")[0],
            )
            papers.append(paper)
        return papers
