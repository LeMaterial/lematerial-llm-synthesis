"""Script that extracts text from a directory called pdf_papers and saves it to a directory called txt_papers."""

import argparse


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
        "--input-path",
        type=str,
        default="data/pdf_papers",
        help="Path to the directory containing the PDFs",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="data/txt_papers/docling",
        help="Path to the directory to save the extracted texts",
    )
    parser.add_argument(
        "--process",
        type=PDFExtractorEnum,
        choices=list(PDFExtractorEnum),
        default="docling",
        help="Extraction process to use (default: 'docling')",
    )
    args = parser.parse_args()

    input_path = args.input_path
    output_path = args.output_path
    extraction_process = args.process

    file_storage = create_file_storage(
        input_path,
    )
    file_storage.create_dir(output_path)
    pdf_extractor = create_pdf_extractor(extraction_process)

    pipeline = ProcessPDFFolderPipeline(
        file_storage=file_storage,
        pdf_extractor=pdf_extractor,
        input_dir=input_path,
        output_dir=output_path,
    )

    pipeline.run()
