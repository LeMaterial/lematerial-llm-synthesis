from gcsfs import GCSFileSystem

from llm_synthesis.services.storage.base_file_storage import BaseFileStorage


class GCSFileStorage(BaseFileStorage):
    """Google Cloud Storage file storage service."""

    def __init__(self):
        self.fs = GCSFileSystem()

    def read_bytes(self, file_path: str) -> bytes:
        """Read a PDF file from GCS and return its content as bytes."""
        with self.fs.open(file_path, "rb") as file:
            return file.read()

    def read_text(self, file_path: str) -> str:
        """Read text data from a file in GCS and return it as a string."""
        with self.fs.open(file_path, "r", encoding="utf-8") as file:
            return file.read()

    def write_text(self, file_path: str, data: str) -> None:
        """Write text data to a file in GCS."""
        with self.fs.open(file_path, "w", encoding="utf-8") as file:
            file.write(data)

    def list_files(self, dir: str, extension: str = "pdf") -> list[str]:
        """
        List all files in the specified directory with the given file extension.

        Args:
            dir (str): The directory path where files should be listed.
            extension (str, optional): The file extension to filter by. Defaults to "pdf".

        Returns:
            list[str]: A list of file names matching the specified extension in the given directory.
        """
        return self.fs.glob(f"{dir}/*.{extension}")
