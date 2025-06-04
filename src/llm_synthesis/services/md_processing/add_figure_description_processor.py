import time
from typing import Optional

from llm_synthesis.extraction.figures.figure_parser import EnhancedFigureParser
from llm_synthesis.services.md_processing.base_md_processor import (
    BaseMarkdownProcessor,
)
from llm_synthesis.utils.figure_utils import (
    clean_text_from_images,  # Added for performance optimization
    find_figures_in_markdown,
    insert_figure_description,
    validate_base64_image,
)


class AddFigureDescriptionsProcessor(BaseMarkdownProcessor):
    """
    A processor that adds descriptions to figures in markdown content.

    Attributes:
        figure_description: A string containing the description to be added to figures.
    """

    def __init__(self, context_window: int = 500, delay_between_requests=10.0):
        self.context_window = context_window
        self.delay_between_requests = delay_between_requests
        self.figure_parser = EnhancedFigureParser()

    def process_markdown(
        self, markdown_data: str, extra_markdown_data: Optional[str] = None
    ) -> str:
        """
        Processes the given markdown data by adding figure descriptions.

        Args:
            markdown_data: The markdown data as a string.
            extra_markdown_data: Optional additional markdown data to append.

        Returns:
            The processed markdown data with figure descriptions added.
        """
        print("Finding figures in markdown...")
        figures = find_figures_in_markdown(markdown_data)
        print(f"Found {len(figures)} figures")

        if not figures:
            print("No figures found, returning original markdown")
            return markdown_data

        # PERFORMANCE OPTIMIZATION: Pre-clean text once to avoid repeated cleaning
        clean_main_text = clean_text_from_images(markdown_data)
        clean_extra_text = (
            clean_text_from_images(extra_markdown_data) if extra_markdown_data else ""
        )

        # Generate descriptions for each figure
        enhanced_markdown = markdown_data

        # LOGIC FIX: Process figures in reverse order to avoid position offset issues
        for i, figure_info in enumerate(reversed(figures)):
            time.sleep(self.delay_between_requests)
            figure_num = len(figures) - i
            print(
                f"Processing figure {figure_num}/{len(figures)}: {figure_info.figure_reference}"
            )

            # Validate the image data
            if not validate_base64_image(figure_info.base64_data):
                print(
                    f"  Skipping invalid image data for {figure_info.figure_reference}"
                )
                continue

            # Prepare context for description generation
            caption_context = figure_info.context_before + figure_info.context_after

            try:
                # Generate description using pre-cleaned text
                description = self.figure_parser.describe_figure(
                    publication_text=clean_main_text,  # Use pre-cleaned text
                    figure_base64=figure_info.base64_data,
                    caption_context=caption_context,
                    figure_position_info=figure_info.figure_reference,
                    si_text=clean_extra_text,  # Use pre-cleaned text
                )

                print(f"  Generated description: {description[:100]}...")

                # Insert description into markdown (no offset adjustment needed in reverse order)
                enhanced_markdown = insert_figure_description(
                    enhanced_markdown, figure_info, description
                )

            except Exception as e:
                print(f"  Error processing {figure_info.figure_reference}: {str(e)}")
                continue

        return enhanced_markdown
