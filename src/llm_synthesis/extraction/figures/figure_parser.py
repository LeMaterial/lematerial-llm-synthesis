import dspy

from llm_synthesis.extraction.figures.signatures import (
    FigureDescriptionSignature,
    FigureExtractionSignature,
)
from llm_synthesis.utils.figure_utils import clean_text_from_images


class FigureParser(dspy.Module):
    """Parser for the figure extraction."""

    def __init__(self):
        self.predict = dspy.Predict(FigureExtractionSignature)

    def __call__(
        self,
        publication_text: str,
        si_text: str,
        figure_bs64: str,
    ) -> str:
        return self.predict(
            publication_text=publication_text,
            si_text=si_text,
            figure_bs64=figure_bs64,
        )


class EnhancedFigureParser(dspy.Module):
    """
    Enhanced parser for figure extraction and description generation.

    This module analyzes figures in research papers and generates detailed
    scientific descriptions that are then integrated into the markdown output.
    """

    def __init__(self):
        self.predict = dspy.Predict(FigureDescriptionSignature)

    def describe_figure(
        self,
        publication_text: str,
        figure_base64: str,
        caption_context: str,
        figure_position_info: str,
        si_text: str | None = None,
    ) -> str:
        """
        Generate a detailed description for a single figure.

        Args:
            publication_text: Main text of the research paper
            figure_base64: Base64 encoded figure image
            caption_context: Text context around the figure position
            figure_position_info: Figure identifier (e.g., "Figure 2",
                "Fig. 3a")
            si_text: Optional supporting information text

        Returns:
            Detailed scientific description of the figure, or
            "NON_SCIENTIFIC_FIGURE" for non-scientific images
        """
        si_text = si_text or ""

        # Clean base64 images from context text to reduce token usage
        clean_publication_text = clean_text_from_images(publication_text)
        clean_si_text = clean_text_from_images(si_text)
        clean_caption_context = clean_text_from_images(caption_context)

        result = self.predict(
            publication_text=clean_publication_text,
            si_text=clean_si_text,
            figure_base64=figure_base64,
            caption_context=clean_caption_context,
            figure_position_info=figure_position_info,
        )

        return result.figure_description.strip()

    def __call__(
        self,
        publication_text: str,
        figure_base64: str,
        caption_context: str,
        figure_position_info: str,
        si_text: str | None = None,
    ) -> str:
        """Compatibility method for the existing signature interface."""
        return self.describe_figure(
            publication_text=publication_text,
            figure_base64=figure_base64,
            caption_context=caption_context,
            figure_position_info=figure_position_info,
            si_text=si_text,
        )
