import base64
import re
from io import BytesIO

from PIL import Image

from llm_synthesis.models.figure import FigureInfo


def extract_figure_context(
    text: str, figure_position: int, context_window: int = 500
) -> tuple[str, str]:
    """
    Extract context before and after a figure position.

    Args:
        text: Full markdown text
        figure_position: Character position of the figure
        context_window: Number of characters to extract before and after

    Returns:
        Tuple of (context_before, context_after)
    """
    start = max(0, figure_position - context_window)
    end = min(len(text), figure_position + context_window)

    context_before = text[start:figure_position]
    context_after = text[figure_position:end]

    return context_before, context_after


def find_figure_reference(context_before: str, context_after: str) -> str:
    """
    Find figure reference (e.g., "Figure 2", "Fig. 3a") in the context.

    Args:
        context_before: Text before the figure
        context_after: Text after the figure

    Returns:
        Figure reference string or "Unknown Figure"
    """
    # Common patterns for figure references
    patterns = [
        r"Figure\s+(\d+[a-z]?)",
        r"Fig\.?\s+(\d+[a-z]?)",
        r"Scheme\s+(\d+[a-z]?)",
        r"Chart\s+(\d+[a-z]?)",
        r"Graph\s+(\d+[a-z]?)",
        r"Image\s+(\d+[a-z]?)",
    ]

    # Search in context_after first (usually where captions are)
    search_text = context_after + " " + context_before

    for pattern in patterns:
        match = re.search(pattern, search_text, re.IGNORECASE)
        if match:
            return match.group(0)

    return "Unknown Figure"


def extract_base64_from_data_uri(data_uri: str) -> str:
    """
    Extract base64 data from a data URI.

    Args:
        data_uri: Data URI string (e.g., "data:image/jpeg;base64,...")

    Returns:
        Base64 encoded string
    """
    if data_uri.startswith("data:"):
        # Split on comma and take the second part (base64 data)
        parts = data_uri.split(",", 1)
        if len(parts) == 2:
            return parts[1]
    return data_uri


def find_figures_in_markdown(markdown_text: str) -> list[FigureInfo]:
    """
    Find all embedded base64 figures in markdown text.

    Args:
        markdown_text: Markdown text with embedded figures

    Returns:
        List of FigureInfo objects containing figure data and context
    """
    figures = []

    # Pattern to match markdown images with data URIs
    pattern = r"!\[([^\]]*)\]\((data:image/[^)]+)\)"

    for match in re.finditer(pattern, markdown_text):
        alt_text = match.group(1)
        data_uri = match.group(2)
        position = match.start()

        # Extract context around the figure
        context_before, context_after = extract_figure_context(
            markdown_text, position, context_window=500
        )

        # Find figure reference
        figure_reference = find_figure_reference(context_before, context_after)

        # Extract base64 data
        base64_data = extract_base64_from_data_uri(data_uri)

        figure_info = FigureInfo(
            base64_data=base64_data,
            alt_text=alt_text,
            position=position,
            context_before=context_before,
            context_after=context_after,
            figure_reference=figure_reference,
        )

        figures.append(figure_info)

    return figures


def insert_figure_description(
    markdown_text: str, figure_info: FigureInfo, description: str
) -> str:
    """
    Insert figure description into markdown text after the figure.

    Args:
        markdown_text: Original markdown text
        figure_info: Information about the figure
        description: Generated description to insert

    Returns:
        Modified markdown text with description inserted
    """
    if description == "NON_SCIENTIFIC_FIGURE":
        return markdown_text

    # Find the end of the figure markdown
    pattern = r"!\[([^\]]*)\]\((data:image/[^)]+)\)"
    match = re.search(pattern, markdown_text[figure_info.position :])

    if not match:
        return markdown_text

    # Position right after the figure
    insert_position = figure_info.position + match.end()

    # Create the description block
    description_block = (
        f"\n\n**AI-Generated Figure Description:** {description}\n"
    )

    # Insert the description
    modified_text = (
        markdown_text[:insert_position]
        + description_block
        + markdown_text[insert_position:]
    )

    return modified_text


def validate_base64_image(base64_data: str) -> bool:
    """
    Validate if base64 data represents a valid image.

    Args:
        base64_data: Base64 encoded image data

    Returns:
        True if valid image data, False otherwise
    """
    try:
        # Try to decode the base64 data
        decoded = base64.b64decode(base64_data)

        # Check for common image file signatures
        if decoded.startswith(b"\xff\xd8\xff"):  # JPEG
            return True
        elif decoded.startswith(b"\x89PNG\r\n\x1a\n"):  # PNG
            return True
        elif decoded.startswith(b"GIF8"):  # GIF
            return True
        elif decoded.startswith(b"RIFF") and b"WEBP" in decoded[:12]:  # WebP
            return True

        # If we can't identify the format, assume it might be valid
        return len(decoded) > 50  # Minimum size check

    except Exception:
        return False


def clean_text_from_images(text: str) -> str:
    """
    Remove base64 image data from text to reduce token count while
    preserving structure.

    Args:
        text: Markdown text containing embedded base64 images

    Returns:
        Cleaned text with images replaced by simple placeholders
    """
    # Pattern to match markdown images with data URIs
    pattern = r"!\[([^\]]*)\]\(data:image/[^)]+\)"

    # Replace with simple placeholder that preserves the figure reference
    def replacement(match):
        alt_text = match.group(1)
        return f"![{alt_text}](placeholder_image)"

    cleaned_text = re.sub(pattern, replacement, text)
    return cleaned_text


def base64_to_image(base64_data: str) -> Image.Image:
    """
    Convert base64 data to an image.

    Args:
        base64_data: Base64 encoded image data

    Returns:
        PIL Image object
    """
    return Image.open(BytesIO(base64.b64decode(base64_data)))
