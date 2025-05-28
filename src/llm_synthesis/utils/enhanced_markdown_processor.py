import time
from pathlib import Path
from typing import List, Optional

from llm_synthesis.extraction.figures.figure_parser import EnhancedFigureParser
from llm_synthesis.utils.dspy_utils import configure_dspy
from llm_synthesis.utils.figure_utils import (
    FigureInfo,
    find_figures_in_markdown,
    insert_figure_description,
    validate_base64_image,
)
from llm_synthesis.utils.parse_utils import extract_markdown


class EnhancedMarkdownProcessor:
    """
    Enhanced markdown processor that extracts figures and generates descriptions.

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
        si_pdf_path: Optional[str] = None,
        engine: str = "mistral",
        root_dir: Optional[str] = None,
        save_output: bool = True,
        context_window: int = 500,
        delay_between_requests=10.0,
        **kwargs,
    ) -> str:
        """
        Process a research paper PDF and generate enhanced markdown with figure descriptions.

        Args:
            pdf_path: Path to the main paper PDF
            si_pdf_path: Optional path to supporting information PDF
            engine: Extraction engine ("mistral" or "docling")
            root_dir: Root directory for outputs
            save_output: Whether to save the enhanced markdown
            context_window: Context window size around figures
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

        # Generate descriptions for each figure
        enhanced_markdown = main_markdown
        offset = 0  # Track text length changes as we insert descriptions

        for i, figure_info in enumerate(figures):
            time.sleep(delay_between_requests)
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
                    publication_text=main_markdown,  # Use text without base64 for context
                    figure_base64=figure_info.base64_data,
                    caption_context=caption_context,
                    figure_position_info=figure_info.figure_reference,
                    si_text=si_markdown,
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

        # Save enhanced markdown if requested
        if save_output and root_dir:
            output_path = self._save_enhanced_markdown(
                enhanced_markdown, pdf_path, engine, root_dir
            )
            print(f"Saved enhanced markdown to: {output_path}")

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
        pdf_paths: List[str],
        si_pdf_paths: Optional[List[str]] = None,
        engine: str = "mistral",
        root_dir: Optional[str] = None,
        **kwargs,
    ) -> List[str]:
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
                print(f"Error processing {pdf_path}: {str(e)}")
                results.append("")

        return results


# Convenience function for easy usage
def process_paper_with_figure_descriptions(
    pdf_path: str,
    si_pdf_path: Optional[str] = None,
    engine: str = "mistral",
    llm_name: str = "gpt-4o",
    model_kwargs: Optional[dict] = {"temperature": 0.1, "max_tokens": 2000},
    root_dir: Optional[str] = None,
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

    # Configure DSPy with the specified LLM
    configure_dspy(llm_name, model_kwargs)

    # Process the paper
    processor = EnhancedMarkdownProcessor()
    return processor.process_paper_with_descriptions(
        pdf_path=pdf_path,
        si_pdf_path=si_pdf_path,
        engine=engine,
        root_dir=root_dir,
        **kwargs,
    )
