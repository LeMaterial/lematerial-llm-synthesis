import time
from pathlib import Path

from llm_synthesis.extraction.figures.figure_parser import EnhancedFigureParser
from llm_synthesis.utils.dspy_utils import configure_dspy
from llm_synthesis.utils.figure_utils import (
    clean_text_from_images,  # Added for performance optimization
    find_figures_in_markdown,
    insert_figure_description,
    validate_base64_image,
)
from llm_synthesis.utils.parse_utils import extract_markdown


class EnhancedMarkdownProcessor:
    """
    Enhanced markdown processor that extracts figures and generates
    descriptions.

    This class orchestrates the entire process of:
    1. Extracting markdown from PDF(s)
    2. Finding embedded figures
    3. Generating detailed descriptions for each figure
    4. Inserting descriptions into the markdown output
    """

    def __init__(self):
        self.figure_parser = EnhancedFigureParser()

    def process_paper_with_descriptions(
        self,
        pdf_path: str,
        si_pdf_path: str | None = None,
        engine: str = "mistral",
        root_dir: str | None = None,
        save_output: bool = True,
        delay_between_requests: float = 10.0,
        **kwargs,
    ) -> str:
        """
        Process a research paper PDF and generate enhanced markdown with
        figure descriptions.

        Args:
            pdf_path: Path to the main paper PDF
            si_pdf_path: Optional path to supporting information PDF
            engine: Extraction engine ("mistral" or "docling")
            root_dir: Root directory for outputs
            save_output: Whether to save the enhanced markdown
            delay_between_requests: Delay between API calls in seconds
            **kwargs: Additional arguments for extract_markdown

        Returns:
            Enhanced markdown text with figure descriptions
        """
        print(f"Processing paper: {pdf_path}")

        # Extract markdown from main paper
        print("Extracting main paper markdown...")
        main_markdown = extract_markdown(
            pdf_path=pdf_path,
            engine=engine,
            image_mode="embedded",  # We need embedded images for analysis
            root_dir=root_dir,
            save_markdown=False,  # We'll save the enhanced version
            **kwargs,
        )

        # Extract supporting information if provided
        si_markdown = ""
        if si_pdf_path:
            print("Extracting supporting information markdown...")
            si_markdown = extract_markdown(
                pdf_path=si_pdf_path,
                engine=engine,
                image_mode="embedded",
                root_dir=root_dir,
                save_markdown=False,
                **kwargs,
            )

        # Find all figures in the main markdown
        print("Finding figures in markdown...")
        figures = find_figures_in_markdown(main_markdown)
        print(f"Found {len(figures)} figures")

        if not figures:
            print("No figures found, returning original markdown")
            return main_markdown

        # PERFORMANCE OPTIMIZATION: Pre-clean text once to avoid repeated
        # cleaning
        clean_main_text = clean_text_from_images(main_markdown)
        clean_si_text = (
            clean_text_from_images(si_markdown) if si_markdown else ""
        )

        # Generate descriptions for each figure
        enhanced_markdown = main_markdown

        # LOGIC FIX: Process figures in reverse order to avoid position offset
        # issues
        for i, figure_info in enumerate(reversed(figures)):
            if delay_between_requests > 0:
                time.sleep(delay_between_requests)

            figure_num = len(figures) - i
            print(
                f"Processing figure {figure_num}/{len(figures)}: "
                f"{figure_info.figure_reference}"
            )

            # Validate the image data
            if not validate_base64_image(figure_info.base64_data):
                print(
                    f"  Skipping invalid image data for "
                    f"{figure_info.figure_reference}"
                )
                continue

            # Prepare context for description generation (also clean it)
            caption_context = clean_text_from_images(
                figure_info.context_before + figure_info.context_after
            )

            try:
                # Generate description using pre-cleaned text
                description = self.figure_parser.describe_figure(
                    publication_text=clean_main_text,  # Use pre-cleaned text
                    figure_base64=figure_info.base64_data,
                    caption_context=caption_context,
                    figure_position_info=figure_info.figure_reference,
                    si_text=clean_si_text,  # Use pre-cleaned text
                )

                print(f"  Generated description: {description[:100]}...")

                # Insert description into markdown (no offset adjustment
                # needed in reverse order)
                enhanced_markdown = insert_figure_description(
                    enhanced_markdown, figure_info, description
                )

            except Exception as e:
                print(
                    f"  Error processing {figure_info.figure_reference!s}: "
                    f"{str(e)!s}"
                )
                continue

        # Save enhanced markdown if requested
        if save_output and root_dir:
            try:
                output_path = self._save_enhanced_markdown(
                    enhanced_markdown, pdf_path, engine, root_dir
                )
                print(f"Saved enhanced markdown to: {output_path}")
            except Exception as e:
                print(f"Error saving enhanced markdown: {str(e)!s}")

        return enhanced_markdown

    def _save_enhanced_markdown(
        self, markdown_content: str, pdf_path: str, engine: str, root_dir: str
    ) -> Path:
        """Save enhanced markdown with figure descriptions."""
        pdf_path = Path(pdf_path)
        root_dir = Path(root_dir)

        # Create output directory structure
        paper_name = pdf_path.stem
        results_dir = root_dir / "results"
        paper_dir = results_dir / paper_name
        engine_dir = paper_dir / engine.lower()

        # Create directories
        engine_dir.mkdir(parents=True, exist_ok=True)

        # Save enhanced markdown
        output_file = engine_dir / f"{paper_name}_enhanced.md"
        output_file.write_text(markdown_content, encoding="utf-8")

        return output_file

    def process_multiple_papers(
        self,
        pdf_paths: list[str],
        si_pdf_paths: list[str] | None = None,
        engine: str = "mistral",
        root_dir: str | None = None,
        **kwargs,
    ) -> list[str]:
        """
        Process multiple papers and generate enhanced markdown for each.

        Args:
            pdf_paths: List of paths to paper PDFs
            si_pdf_paths: Optional list of supporting information PDFs
            engine: Extraction engine
            root_dir: Root directory for outputs
            **kwargs: Additional arguments

        Returns:
            List of enhanced markdown strings
        """
        if si_pdf_paths is None:
            si_pdf_paths = [None] * len(pdf_paths)

        if len(si_pdf_paths) != len(pdf_paths):
            raise ValueError(
                "Length of si_pdf_paths must match pdf_paths if provided"
            )

        results = []
        for i, (pdf_path, si_path) in enumerate(zip(pdf_paths, si_pdf_paths)):
            print(f"\n=== Processing paper {i + 1}/{len(pdf_paths)} ===")
            try:
                enhanced_md = self.process_paper_with_descriptions(
                    pdf_path=pdf_path,
                    si_pdf_path=si_path,
                    engine=engine,
                    root_dir=root_dir,
                    **kwargs,
                )
                results.append(enhanced_md)
            except Exception as e:
                print(f"Error processing {pdf_path!s}: {str(e)!s}")
                results.append("")

        return results


# Convenience function for easy usage
def process_paper_with_figure_descriptions(
    pdf_path: str,
    si_pdf_path: str | None = None,
    engine: str = "mistral",
    llm_name: str = "gpt-4o",
    model_kwargs: dict | None = None,
    root_dir: str | None = None,
    save_output: bool = False,
    **kwargs,
) -> str:
    """
    Convenience function to process a paper with figure descriptions.

    Args:
        pdf_path: Path to the main paper PDF
        si_pdf_path: Optional path to supporting information PDF
        engine: Extraction engine ("mistral" or "docling")
        llm_name: LLM to use for description generation
        model_kwargs: Model configuration parameters
        root_dir: Root directory for outputs
        **kwargs: Additional arguments for markdown extraction

    Returns:
        Enhanced markdown with figure descriptions
    """
    if model_kwargs is None:
        model_kwargs = {"temperature": 0.1, "max_tokens": 2000}

    # Configure DSPy with the specified LLM
    configure_dspy(llm_name, model_kwargs)

    # Process the paper
    processor = EnhancedMarkdownProcessor()
    return processor.process_paper_with_descriptions(
        pdf_path=pdf_path,
        si_pdf_path=si_pdf_path,
        engine=engine,
        root_dir=root_dir,
        save_output=save_output,
        **kwargs,
    )
