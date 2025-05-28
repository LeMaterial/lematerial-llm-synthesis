import os
import json
import base64
import mimetypes
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import ImageRefMode
from mistralai import Mistral


def _create_output_structure(
    pdf_path: str, engine: str, root_dir: str = None
) -> tuple[Path, Path, Path]:
    """
    Create organized output directory structure.

    Structure:
    root/
    ├── results/
    │   └── {paper_name}/
    │       ├── mistral/
    │       │   ├── {paper_name}.md
    │       │   └── figures/
    │       └── docling/
    │           ├── {paper_name}.md
    │           └── figures/

    Args:
        pdf_path: Path to the source PDF
        engine: Processing engine name ("mistral" or "docling")
        root_dir: Root directory (defaults to PDF's parent directory)

    Returns:
        Tuple of (paper_dir, engine_dir, figures_dir)
    """
    pdf_path = Path(pdf_path)

    # Use PDF's parent directory as root if not specified
    if root_dir is None:
        root_dir = pdf_path.parent
    else:
        root_dir = Path(root_dir)

    # Extract paper name (without extension)
    paper_name = pdf_path.stem

    # Create directory structure
    results_dir = root_dir / "results"
    paper_dir = results_dir / paper_name
    engine_dir = paper_dir / engine.lower()
    figures_dir = engine_dir / "figures"

    # Create all directories
    figures_dir.mkdir(parents=True, exist_ok=True)

    return paper_dir, engine_dir, figures_dir


def mistral_markdown(
    pdf_path: str,
    structured: bool = False,
    image_mode: str = "embedded",
    root_dir: str = None,
    save_markdown: bool = True,
) -> str:
    """
    Extract markdown from PDF using Mistral OCR with organized file structure.

    Args:
        pdf_path: Path to the PDF file
        structured: Whether to return structured JSON instead of markdown
        image_mode: "embedded", "referenced", or "placeholder"
        root_dir: Root directory for outputs (defaults to PDF's parent)
        save_markdown: Whether to save markdown file to disk

    Returns:
        Markdown string with images handled according to image_mode
    """
    pdf_path = Path(pdf_path)

    client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
    data_uri = (
        "data:application/pdf;base64,"
        + base64.b64encode(pdf_path.read_bytes()).decode()
    )

    # Always request images from Mistral API unless placeholder mode
    include_images = image_mode != "placeholder"

    resp = client.ocr.process(
        document={"type": "document_url", "document_url": data_uri},
        model="mistral-ocr-latest",
        include_image_base64=include_images,
    )

    if structured:
        return json.dumps(resp.to_dict(), indent=2)

    # Set up output directories for referenced mode
    if image_mode == "referenced":
        paper_dir, engine_dir, figures_dir = _create_output_structure(
            pdf_path, "mistral", root_dir
        )

    pages_out = []
    for page_idx, page in enumerate(resp.pages):
        md = page.markdown

        if image_mode == "placeholder":
            # Replace all image references with placeholder text
            if getattr(page, "images", None):
                for img in page.images:
                    key = getattr(img, "file_name", None) or getattr(img, "id", "")
                    if key:
                        md = md.replace(f"![{key}]({key})", "![Image placeholder]")

        elif image_mode == "embedded":
            # Original behavior: embed as data URIs
            if getattr(page, "images", None):
                uri_for = {}
                for img in page.images:
                    key = getattr(img, "file_name", None) or getattr(img, "id", "")
                    raw = getattr(img, "data_uri", None) or getattr(
                        img, "image_base64", ""
                    )
                    if not raw:
                        continue
                    if not raw.startswith("data:"):
                        mime = mimetypes.guess_type(key)[0] or "image/jpeg"
                        raw = f"data:{mime};base64,{raw}"
                    uri_for[key] = raw

                for key, uri in uri_for.items():
                    md = md.replace(f"![{key}]({key})", f"![fig]({uri})")

        elif image_mode == "referenced":
            # Save images to organized file structure
            if getattr(page, "images", None):
                for img_idx, img in enumerate(page.images):
                    key = getattr(img, "file_name", None) or getattr(img, "id", "")
                    raw = getattr(img, "data_uri", None) or getattr(
                        img, "image_base64", ""
                    )

                    if not raw:
                        continue

                    # Generate descriptive filename
                    if key and key.strip():
                        # Clean the key to be filesystem-safe
                        safe_key = "".join(c for c in key if c.isalnum() or c in "._-")
                        filename = f"page_{page_idx + 1:02d}_{safe_key}"
                    else:
                        filename = f"page_{page_idx + 1:02d}_img_{img_idx + 1:02d}"

                    # Ensure proper extension
                    if not Path(filename).suffix:
                        filename += ".jpg"

                    # Full path to save image
                    image_file_path = figures_dir / filename

                    # Extract and save image data
                    if raw.startswith("data:"):
                        base64_data = raw.split(",", 1)[1]
                    else:
                        base64_data = raw

                    # Save image file
                    image_file_path.write_bytes(base64.b64decode(base64_data))

                    # Create relative path from markdown file to image
                    # Markdown will be saved in engine_dir, images are in figures_dir
                    relative_path = Path("figures") / filename
                    md = md.replace(f"![{key}]({key})", f"![fig]({relative_path})")

        else:
            raise ValueError(
                "image_mode must be one of: embedded | referenced | placeholder"
            )

        pages_out.append(md)

    markdown_content = "\n\n".join(pages_out)

    # Save markdown file if requested and not in embedded mode with no structure needed
    if save_markdown and (image_mode == "referenced" or root_dir is not None):
        if image_mode != "referenced":
            # Create structure even for embedded/placeholder modes if root_dir specified
            paper_dir, engine_dir, figures_dir = _create_output_structure(
                pdf_path, "mistral", root_dir
            )

        markdown_file_path = engine_dir / f"{pdf_path.stem}.md"
        markdown_file_path.write_text(markdown_content, encoding="utf-8")
        print(f"Saved markdown to: {markdown_file_path}")

        if image_mode == "referenced":
            print(
                f"Saved {len([f for f in figures_dir.iterdir() if f.is_file()])} images to: {figures_dir}"
            )

    return markdown_content


def docling_markdown(
    pdf_path: str,
    pipeline: str = "standard",
    table_mode: str = "accurate",
    image_mode: str = "embedded",
    add_page_images: bool = False,
    use_gpu: bool = True,
    scale: float = 2.0,
    format: str = "markdown",
    root_dir: str = None,
    save_markdown: bool = True,
) -> str:
    """
    Convert a PDF to markdown with Docling using organized file structure.
    """
    pdf_path = Path(pdf_path)

    # Map image mode names to Docling enums
    mode_map = {
        "embedded": ImageRefMode.EMBEDDED,
        "embed": ImageRefMode.EMBEDDED,
        "referenced": ImageRefMode.REFERENCED,
        "reference": ImageRefMode.REFERENCED,
        "placeholder": ImageRefMode.PLACEHOLDER,
    }
    try:
        image_mode_enum = mode_map[image_mode.lower()]
    except KeyError as e:
        raise ValueError(
            "image_mode must be one of: embedded | referenced | placeholder"
        ) from e

    # Set up output directories for referenced mode
    if image_mode.lower() in ["referenced", "reference"]:
        paper_dir, engine_dir, figures_dir = _create_output_structure(
            pdf_path, "docling", root_dir
        )

    opts = PdfPipelineOptions(
        pipeline=pipeline,
        table_mode=table_mode,
        generate_picture_images=True,
        generate_page_images=add_page_images,
        images_scale=scale,
        ocr=True,
        batch_size=4 if use_gpu else 1,
    )
    conv = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
    )

    result = conv.convert(pdf_path)
    doc = result.document

    if format == "markdown":
        content = doc.export_to_markdown(image_mode=image_mode_enum)
    elif format == "doctags":
        content = doc.export_to_doctags()
    elif format == "json":
        content = doc.export_to_dict()
    elif format == "tokens":
        content = doc.export_to_document_tokens()
    else:
        raise ValueError("format must be one of: markdown | doctags | json | tokens")

    # Save markdown file if requested
    if (
        save_markdown
        and format == "markdown"
        and (image_mode.lower() in ["referenced", "reference"] or root_dir is not None)
    ):
        if image_mode.lower() not in ["referenced", "reference"]:
            # Create structure even for embedded/placeholder modes if root_dir specified
            paper_dir, engine_dir, figures_dir = _create_output_structure(
                pdf_path, "docling", root_dir
            )

        markdown_file_path = engine_dir / f"{pdf_path.stem}.md"
        markdown_file_path.write_text(content, encoding="utf-8")
        print(f"Saved markdown to: {markdown_file_path}")

        if image_mode.lower() in ["referenced", "reference"]:
            print(f"Images saved to: {figures_dir}")

    return content


def extract_markdown(
    pdf_path: str,
    engine: str = "mistral",
    image_mode: str = "embedded",
    root_dir: str = None,
    save_markdown: bool = True,
    **kwargs,
) -> str:
    """
    Extract markdown from a PDF using the specified engine with organized output structure.

    Args:
        pdf_path: Path to the PDF file
        engine: "mistral" or "docling"
        image_mode: "embedded", "referenced", or "placeholder"
        root_dir: Root directory for organized outputs (defaults to PDF's parent)
        save_markdown: Whether to save markdown and create directory structure
        **kwargs: Additional arguments passed to the specific engine

    Returns:
        Markdown content as string

    Example directory structure created:
        root_dir/
        ├── results/
        │   └── paper_name/
        │       ├── mistral/
        │       │   ├── paper_name.md
        │       │   └── figures/
        │       │       ├── page_01_fig1.jpg
        │       │       └── page_02_table1.png
        │       └── docling/
        │           ├── paper_name.md
        │           └── figures/
    """
    engine = engine.lower()
    if engine == "mistral":
        return mistral_markdown(
            pdf_path,
            image_mode=image_mode,
            root_dir=root_dir,
            save_markdown=save_markdown,
            **kwargs,
        )
    elif engine == "docling":
        return docling_markdown(
            pdf_path,
            image_mode=image_mode,
            root_dir=root_dir,
            save_markdown=save_markdown,
            **kwargs,
        )
    else:
        raise ValueError("engine must be mistral | docling")


def change_extension(file_path: str, new_extension: str) -> str:
    """
    Change the extension of a file.
    """
    base_name, _ = os.path.splitext(file_path)
    new_file_path = base_name + "." + new_extension
    return new_file_path
