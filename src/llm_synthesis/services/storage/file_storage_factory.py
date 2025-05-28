from llm_synthesis.services.storage.base_file_storage import BaseFileStorage
from llm_synthesis.services.storage.gcs_file_storage import GCSFileStorage
from llm_synthesis.services.storage.local_file_storage import LocalFileStorage


def create_file_storage(base_path: str) -> BaseFileStorage:
    """
    Factory function to create a file storage service based on the settings.

    Returns:
        A file storage instance.
    """
    if base_path.startswith("gs://"):
        return GCSFileStorage()
    else:
        return LocalFileStorage()
