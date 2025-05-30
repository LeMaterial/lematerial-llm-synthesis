from abc import ABC, abstractmethod


class BaseFileStorage(ABC):

    @abstractmethod
    def create_dir(self, dir: str) -> None:
        """Create a directory."""
        pass

    @abstractmethod
    def read_bytes(self, filepath: str) -> bytes:
        """Read a file and return its content as bytes."""
        pass

    @abstractmethod
    def read_text(self, filepath: str) -> str:
        """Read a file and return its content as string."""
        pass

    @abstractmethod
    def write_text(self, filepath: str, data: str) -> None:
        """Write string data to a file."""
        pass

    @abstractmethod
    def list_files(self, dir: str, extension: str = "pdf") -> list[str]:
        """
        List all files in the specified directory with the given file extension.

        Args:
            dir (str): The directory path where files should be listed.
            extension (str, optional): The file extension to filter by. Defaults to "pdf".

        Returns:
            list[str]: A list of file names matching the specified extension in the given directory.
        """
        pass
