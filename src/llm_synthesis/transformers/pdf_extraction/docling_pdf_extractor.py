import io

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.io import DocumentStream

from llm_synthesis.transformers.pdf_extraction.base import PdfExtractorInterface


class DoclingPDFExtractor(PdfExtractorInterface):
    """
    An extractor for extracting content from PDF files using the Docling library.

    This class provides functionality to convert PDF files into various formats
    such as Markdown, doctags, JSON, or tokens. It supports different modes for
    handling images within the PDF, including embedding, referencing, or using
    placeholders. The extractor can be configured with various options such as
    pipeline type, table extraction mode, GPU usage, and scaling.

    Attributes:
        pipeline (str): The pipeline to use for PDF processing (default: "standard").
        table_mode (str): The mode for table extraction (default: "accurate").
        add_page_images (bool): Whether to include page images in the output (default: False).
        use_gpu (bool): Whether to use GPU for processing (default: True).
        scale (float): The scaling factor for images (default: 2.0).
        format (str): The output format. Options are "markdown", "doctags", "json", or "tokens" (default: "markdown").

    Methods:

        extract_to_markdown(pdf_data: bytes) -> str:
            Converts a PDF file to Markdown format. Supports different image modes
            and raises a ValueError if an invalid image mode is provided.
    """

    def __init__(
        self,
        pipeline: str = "standard",
        table_mode: str = "accurate",
        add_page_images: bool = False,
        use_gpu: bool = True,
        scale: float = 2.0,
        format: str = "markdown",
    ):
        self.pipeline = pipeline
        self.table_mode = table_mode
        self.add_page_images = add_page_images
        self.use_gpu = use_gpu
        self.scale = scale
        self.format = format

    def extract(self, input: bytes) -> str:
        """
        Extracts text and figures from a PDF and returns them as markdown with embedded figures.

        Args:
            pdf_data: The PDF data as bytes.

        Returns:
            The extracted text as markdown with embedded figures.
        """
        opts = PdfPipelineOptions(
            pipeline=self.pipeline,
            table_mode=self.table_mode,
            generate_picture_images=True,
            generate_page_images=self.add_page_images,
            images_scale=self.scale,
            ocr=True,
            batch_size=4 if self.use_gpu else 1,
        )
        conv = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
        )
        result = conv.convert(DocumentStream(name="pdf", stream=io.BytesIO(input)))
        doc = result.document

        return doc.export_to_markdown(image_mode="embedded")
