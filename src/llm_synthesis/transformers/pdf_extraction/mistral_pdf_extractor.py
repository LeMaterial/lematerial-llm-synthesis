import base64
import json
import logging
import mimetypes
import os

from mistralai import Mistral

from llm_synthesis.transformers.pdf_extraction.base import PdfExtractorInterface

LOGGER = logging.getLogger(__name__)


class MistralPDFExtractor(PdfExtractorInterface):
    """
    A PDF extractor that uses the Mistral OCR API to extract content from PDF files
    and optionally convert it to Markdown format. This extractor supports embedding
    images as data URIs and can return structured JSON output if required.

    Attributes:
        structured (bool): Determines whether the output should be structured JSON.
        embed_images (bool): Indicates whether images should be embedded as data URIs in the Markdown output.
        mistral_api_key (str): The API key for authenticating with the Mistral OCR API.
                                If not provided, it will be fetched from the environment variable `MISTRAL_API_KEY`.
        mistral_api_client (Mistral): The client instance for interacting with the Mistral OCR API.

    Methods:
        __init__(structured: bool = False, embed_images: bool = True, mistral_api_key: str = None):
            Initializes the MistralPDFExtractor with the given configuration and API key.

        extract_to_markdown(pdf_data: bytes) -> str:
            Extracts content from a PDF file and converts it to Markdown format.
            Optionally embeds images as data URIs and supports structured JSON output.
    """

    def __init__(
        self,
        structured: bool = False,
        mistral_api_key: str = None,
    ):
        self.structured = structured
        self.mistral_api_key = mistral_api_key or os.environ.get("MISTRAL_API_KEY")
        if self.mistral_api_key is None:
            LOGGER.error(
                "MISTRAL_API_KEY is not set. Please provide it as an argument or set it in the environment."
            )
            raise ValueError(
                "MISTRAL_API_KEY must be set either as an argument or in the environment."
            )
        print(self.mistral_api_key)
        self.mistral_api_client = Mistral(api_key=self.mistral_api_key)

    def extract(self, input: bytes) -> str:
        """
        Extracts text and figures from a PDF and returns them as markdown with embedded figures.

        Args:
            pdf_data: The PDF data as bytes.

        Returns:
            The extracted text as markdown with embedded figures.
        """
        data_uri = "data:application/pdf;base64," + base64.b64encode(input).decode()

        resp = self.mistral_api_client.ocr.process(
            document={"type": "document_url", "document_url": data_uri},
            model="mistral-ocr-latest",
            include_image_base64=True,
        )

        if self.structured:  # <-- optional JSON dump
            return json.dumps(resp.to_dict(), indent=2)

        pages_out = []
        for page in resp.pages:
            md = page.markdown

            # build *id*  -> full data-URI
            uri_for: dict[str, str] = {}
            for img in page.images:
                key = getattr(img, "file_name", None) or getattr(img, "id", "")
                raw = getattr(img, "data_uri", None) or getattr(img, "image_base64", "")
                if not raw:
                    continue
                if not raw.startswith("data:"):  # add header if needed
                    mime = mimetypes.guess_type(key)[0] or "image/jpeg"
                    raw = f"data:{mime};base64,{raw}"
                uri_for[key] = raw

            # replace every ![key](key) with the inline image
            for key, uri in uri_for.items():
                md = md.replace(f"![{key}]({key})", f"![fig]({uri})")

            pages_out.append(md)

        return "\n\n".join(pages_out)
