import logging
import os

from llm_synthesis.services.pipelines.base_pipeline import BasePipeline
from llm_synthesis.services.storage.base_file_storage import BaseFileStorage
from llm_synthesis.transformers.pdf_extraction.base import (
    PdfExtractorInterface,
)

logger = logging.getLogger(__name__)


class ProcessPDFFolderPipeline(BasePipeline):
    def __init__(
        self,
        file_storage: BaseFileStorage,
        pdf_extractor: PdfExtractorInterface,
        input_dir: str = "data/pdf_files",
        output_dir: str = "data/txt_files/docling",
    ):
        """
        Initialize the pipeline with a file storage and a PDF extractor.

        Args:
            file_storage (BaseFileStorage): The file storage service to use.
            pdf_extractor (PdfExtractorInterface): The PDF extractor to use.
        """
        self.file_storage = file_storage
        self.pdf_extractor = pdf_extractor
        self.input_dir = input_dir
        self.output_dir = output_dir
        super().__init__()

    def run(self) -> None:
        """
        Run the pipeline to process all PDF files in the specified directory.

        This method reads all PDF files from the specified directory, extracts
        their text, and writes the extracted text to corresponding TXT files
        in the output directory.
        """
        pdf_files = self.file_storage.list_files(
            self.input_dir, extension="pdf"
        )

        for pdf_file in pdf_files:
            pdf_content = self.file_storage.read_bytes(pdf_file)
            extracted_text = (
                self.pdf_extractor.extract_to_markdown_with_figures_embedded(
                    pdf_content
                )
            )
            txt_file = pdf_file.split("/")[-1].replace(".pdf", ".txt")
            self.file_storage.write_text(
                os.path.join(self.output_dir, txt_file), extracted_text
            )
