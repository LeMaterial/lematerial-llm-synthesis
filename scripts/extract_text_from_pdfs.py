"""Script that extracts text from a directory called pdf_papers and saves it to a directory called txt_papers."""

import argparse
import os

from dotenv import load_dotenv

from llm_synthesis.services.pdf_extraction.pdf_extractor_factory import (
    PDFExtractorEnum,
    create_pdf_extractor,
)
from llm_synthesis.services.pipelines.process_pdf_folder_pipeline import (
    ProcessPDFFolderPipeline,
)
from llm_synthesis.services.storage.file_storage_factory import create_file_storage

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract text from PDFs.")
    parser.add_argument(
        "--base-path",
        type=str,
        default="data",
        help="Path to the working directory (default: 'data')",
    )
    parser.add_argument(
        "--process",
        type=PDFExtractorEnum,
        choices=list(PDFExtractorEnum),
        default="docling",
        help="Extraction process to use (default: 'docling')",
    )
    args = parser.parse_args()

    base_path = args.base_path
    extraction_process = args.process

    load_dotenv()
    os.makedirs(os.path.join(base_path, "pdf_files"), exist_ok=True)
    os.makedirs(
        os.path.join(base_path, "txt_files", extraction_process.value), exist_ok=True
    )

    file_storage = create_file_storage(
        base_path,
    )
    pdf_extractor = create_pdf_extractor(extraction_process)

    pipeline = ProcessPDFFolderPipeline(
        file_storage=file_storage,
        pdf_extractor=pdf_extractor,
        input_dir=os.path.join(base_path, "pdf_files"),
        output_dir=os.path.join(base_path, "txt_files", extraction_process.value),
    )

    pipeline.run()
