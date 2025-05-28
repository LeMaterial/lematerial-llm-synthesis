import time
from typing import Optional

from llm_synthesis.services.md_processing.base_md_processor import (
    BaseMarkdownProcessor,
)
from llm_synthesis.extraction.figures.figure_parser import EnhancedFigureParser
from llm_synthesis.utils.figure_utils import (
    FigureInfo,
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

        # Generate descriptions for each figure
        enhanced_markdown = markdown_data
        offset = 0  # Track text length changes as we insert descriptions

        for i, figure_info in enumerate(figures):
            time.sleep(self.delay_between_requests)
            print(
                f"Processing figure {i + 1}/{len(figures)}: {figure_info.figure_reference}"
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
                # Generate description
                description = self.figure_parser.describe_figure(
                    publication_text=markdown_data,  # Use text without base64 for context
                    figure_base64=figure_info.base64_data,
                    caption_context=caption_context,
                    figure_position_info=figure_info.figure_reference,
                    si_text=extra_markdown_data or "",
                )

                print(f"  Generated description: {description[:100]}...")

                # Adjust figure position for the offset
                adjusted_figure_info = FigureInfo(
                    base64_data=figure_info.base64_data,
                    alt_text=figure_info.alt_text,
                    position=figure_info.position + offset,
                    context_before=figure_info.context_before,
                    context_after=figure_info.context_after,
                    figure_reference=figure_info.figure_reference,
                )

                # Insert description into markdown
                old_length = len(enhanced_markdown)
                enhanced_markdown = insert_figure_description(
                    enhanced_markdown, adjusted_figure_info, description
                )
                new_length = len(enhanced_markdown)
                offset += new_length - old_length

            except Exception as e:
                print(f"  Error processing {figure_info.figure_reference}: {str(e)}")
                continue

        return enhanced_markdown
