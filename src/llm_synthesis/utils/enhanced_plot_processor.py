import json
import time
from pathlib import Path

from llm_synthesis.extraction.figures.figure_parser import EnhancedFigureParser
from llm_synthesis.extraction.plots.plot_parser import ComprehensivePlotParser
from llm_synthesis.utils.dspy_utils import configure_dspy
from llm_synthesis.utils.figure_utils import (
    clean_text_from_images,
    find_figures_in_markdown,
    insert_figure_description,
    validate_base64_image,
)
from llm_synthesis.utils.parse_utils import extract_markdown
from llm_synthesis.utils.plot_utils import (
    export_data_to_csv,
    export_data_to_json,
    summarize_extraction_results,
    validate_extracted_data,
)


class EnhancedPlotProcessor:
    """
    Enhanced processor that extracts both figure descriptions and plot data.

    This class orchestrates the process of:
    1. Extracting markdown from PDF(s)
    2. Finding embedded figures
    3. Identifying which figures contain extractable plots
    4. Extracting data points from plots
    5. Generating figure descriptions for non-plot figures
    6. Analyzing extracted plot data
    7. Saving results in multiple formats
    """

    def __init__(self):
        self.plot_parser = ComprehensivePlotParser()
        self.figure_parser = EnhancedFigureParser()

    def process_paper_with_plot_extraction(
        self,
        pdf_path: str,
        si_pdf_path: str | None = None,
        engine: str = "mistral",
        root_dir: str | None = None,
        save_output: bool = True,
        extract_plot_data: bool = True,
        generate_figure_descriptions: bool = True,
        delay_between_requests: float = 10.0,
        export_formats: list[str] = ["csv", "json"],
        **kwargs,
    ) -> dict[str, any]:
        """
        Process a research paper PDF with both plot data extraction and figure
        descriptions.

        Args:
            pdf_path: Path to the main paper PDF
            si_pdf_path: Optional path to supporting information PDF
            engine: Extraction engine ("mistral" or "docling")
            root_dir: Root directory for outputs
            save_output: Whether to save results to files
            extract_plot_data: Whether to extract data from plots
            generate_figure_descriptions: Whether to generate descriptions for
            non-plot figures
            delay_between_requests: Delay between API calls in seconds
            export_formats: List of export formats ("csv", "json", "summary")
            **kwargs: Additional arguments for extract_markdown

        Returns:
            Dictionary containing processing results
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
            return {
                "markdown": main_markdown,
                "figures_processed": 0,
                "plots_extracted": 0,
                "descriptions_added": 0,
            }

        # Pre-clean text once for efficiency
        clean_main_text = clean_text_from_images(main_markdown)
        clean_si_text = (
            clean_text_from_images(si_markdown) if si_markdown else ""
        )

        # Process each figure
        results = {
            "markdown": main_markdown,
            "plot_data": [],
            "figure_descriptions": [],
            "processing_errors": [],
            "figures_processed": 0,
            "plots_extracted": 0,
            "descriptions_added": 0,
        }

        enhanced_markdown = main_markdown

        # Process figures in reverse order to avoid position offset issues
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

            # Prepare context for processing
            caption_context = clean_text_from_images(
                figure_info.context_before + figure_info.context_after
            )

            try:
                # Try plot extraction first if enabled
                plot_extracted = False
                if extract_plot_data:
                    print("  Attempting plot data extraction...")
                    plot_results = self.plot_parser.process_figure(
                        figure_base64=figure_info.base64_data,
                        publication_context=clean_main_text,
                        figure_caption=caption_context,
                        extract_data=True,
                        analyze_data=True,
                    )

                    if plot_results.get("identification", {}).get(
                        "is_extractable", False
                    ):
                        print("  Successfully extracted plot data!")
                        plot_results["figure_reference"] = (
                            figure_info.figure_reference
                        )
                        plot_results["figure_position"] = figure_info.position
                        results["plot_data"].append(plot_results)
                        results["plots_extracted"] += 1
                        plot_extracted = True

                        # Insert plot analysis into markdown if available
                        if "analysis" in plot_results:
                            analysis_text = (
                                f"\n\n**Plot Data Analysis:** "
                                f"{plot_results['analysis']}\n"
                            )
                            enhanced_markdown = self._insert_text_after_figure(
                                enhanced_markdown, figure_info, analysis_text
                            )
                    else:
                        print("  Figure not identified as extractable plot")

                # Generate figure description if not a plot or if descriptions
                # are still wanted
                if generate_figure_descriptions and not plot_extracted:
                    print("  Generating figure description...")
                    description = self.figure_parser.describe_figure(
                        publication_text=clean_main_text,
                        figure_base64=figure_info.base64_data,
                        caption_context=caption_context,
                        figure_position_info=figure_info.figure_reference,
                        si_text=clean_si_text,
                    )

                    if description != "NON_SCIENTIFIC_FIGURE":
                        print(
                            f"  Generated description: {description[:100]}..."
                        )
                        enhanced_markdown = insert_figure_description(
                            enhanced_markdown, figure_info, description
                        )
                        results["descriptions_added"] += 1

                        results["figure_descriptions"].append(
                            {
                                "figure_reference": (
                                    figure_info.figure_reference
                                ),
                                "description": description,
                            }
                        )

                results["figures_processed"] += 1

            except Exception as e:
                error_msg = (
                    f"Error processing {figure_info.figure_reference}: {e!s}"
                )
                print(f"  {error_msg}")
                results["processing_errors"].append(error_msg)
                continue

        results["markdown"] = enhanced_markdown

        # Save results if requested
        if save_output and root_dir:
            try:
                self._save_processing_results(
                    results, pdf_path, engine, root_dir, export_formats
                )
            except Exception as e:
                print(f"Error saving results: {e!s}")
                results["save_error"] = str(e)

        return results

    def _insert_text_after_figure(
        self, markdown_text: str, figure_info, text_to_insert: str
    ) -> str:
        """Insert text after a figure in markdown."""
        # Find the figure pattern
        import re

        pattern = r"!\[([^\]]*)\]\((data:image/[^)]+)\)"
        match = re.search(pattern, markdown_text[figure_info.position :])

        if not match:
            return markdown_text

        # Position right after the figure
        insert_position = figure_info.position + match.end()

        # Insert the text
        modified_text = (
            markdown_text[:insert_position]
            + text_to_insert
            + markdown_text[insert_position:]
        )

        return modified_text

    def _save_processing_results(
        self,
        results: dict[str, any],
        pdf_path: str,
        engine: str,
        root_dir: str,
        export_formats: list[str],
    ) -> None:
        """Save processing results in various formats."""
        pdf_path = Path(pdf_path)
        root_dir = Path(root_dir)

        # Create output directory structure
        paper_name = pdf_path.stem
        results_dir = root_dir / "results"
        paper_dir = results_dir / paper_name
        engine_dir = paper_dir / engine.lower()
        plots_dir = engine_dir / "plot_data"

        # Create directories
        plots_dir.mkdir(parents=True, exist_ok=True)

        # Save enhanced markdown
        enhanced_markdown_path = (
            engine_dir / f"{paper_name}_enhanced_with_plots.md"
        )
        enhanced_markdown_path.write_text(
            results["markdown"], encoding="utf-8"
        )
        print(f"Saved enhanced markdown to: {enhanced_markdown_path}")

        # Save plot data if any was extracted
        if results["plot_data"]:
            # Combine all extracted plot data
            all_extracted_data = []
            all_analyses = []

            for plot_result in results["plot_data"]:
                if "extracted_data" in plot_result:
                    all_extracted_data.extend(plot_result["extracted_data"])
                if "analysis" in plot_result:
                    all_analyses.append(plot_result["analysis"])

            # Export in requested formats
            if "csv" in export_formats and all_extracted_data:
                csv_path = plots_dir / f"{paper_name}_plot_data.csv"
                export_data_to_csv(all_extracted_data, str(csv_path))
                print(f"Saved plot data CSV to: {csv_path}")

            if "json" in export_formats and all_extracted_data:
                json_path = plots_dir / f"{paper_name}_plot_data.json"
                combined_analysis = (
                    "\n\n".join(all_analyses) if all_analyses else None
                )
                export_data_to_json(
                    all_extracted_data, str(json_path), combined_analysis
                )
                print(f"Saved plot data JSON to: {json_path}")

            if "summary" in export_formats and all_extracted_data:
                summary_path = plots_dir / f"{paper_name}_plot_summary.txt"
                combined_analysis = (
                    "\n\n".join(all_analyses) if all_analyses else None
                )
                summary = summarize_extraction_results(
                    all_extracted_data, combined_analysis
                )
                summary_path.write_text(summary, encoding="utf-8")
                print(f"Saved plot summary to: {summary_path}")

            # Save validation results
            if all_extracted_data:
                validation = validate_extracted_data(all_extracted_data)
                validation_path = plots_dir / f"{paper_name}_validation.json"
                with open(validation_path, "w", encoding="utf-8") as f:
                    json.dump(validation, f, indent=2, default=str)
                print(f"Saved validation results to: {validation_path}")

        # Save processing summary
        summary_data = {
            "processing_summary": {
                "figures_found": results["figures_processed"],
                "plots_extracted": results["plots_extracted"],
                "descriptions_added": results["descriptions_added"],
                "errors": results["processing_errors"],
            },
            "plot_results": [
                {
                    "figure_reference": pr["figure_reference"],
                    "plot_type": pr.get("identification", {}).get(
                        "plot_type", "unknown"
                    ),
                    "subplot_count": pr.get("identification", {}).get(
                        "subplot_count", 0
                    ),
                    "extraction_successful": "extracted_data" in pr,
                }
                for pr in results["plot_data"]
            ],
        }

        summary_path = engine_dir / f"{paper_name}_processing_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2)
        print(f"Saved processing summary to: {summary_path}")


# Convenience function for easy usage
def process_paper_with_plot_extraction(
    pdf_path: str,
    si_pdf_path: str | None = None,
    engine: str = "mistral",
    llm_name: str = "gpt-4o",
    model_kwargs: dict | None = None,
    root_dir: str | None = None,
    save_output: bool = True,
    extract_plot_data: bool = True,
    generate_figure_descriptions: bool = True,
    **kwargs,
) -> dict[str, any]:
    """
    Convenience function to process a paper with both plot extraction and
    figure descriptions.

    Args:
        pdf_path: Path to the main paper PDF
        si_pdf_path: Optional path to supporting information PDF
        engine: Extraction engine ("mistral" or "docling")
        llm_name: LLM to use for processing
        model_kwargs: Model configuration parameters
        root_dir: Root directory for outputs
        save_output: Whether to save results to files
        extract_plot_data: Whether to extract data from plots
        generate_figure_descriptions: Whether to generate descriptions for
        non-plot figures
        **kwargs: Additional arguments for markdown extraction

    Returns:
        Dictionary containing all processing results
    """
    if model_kwargs is None:
        model_kwargs = {"temperature": 0.1, "max_tokens": 3000}

    # Configure DSPy with the specified LLM
    configure_dspy(llm_name, model_kwargs)

    # Process the paper
    processor = EnhancedPlotProcessor()
    return processor.process_paper_with_plot_extraction(
        pdf_path=pdf_path,
        si_pdf_path=si_pdf_path,
        engine=engine,
        root_dir=root_dir,
        save_output=save_output,
        extract_plot_data=extract_plot_data,
        generate_figure_descriptions=generate_figure_descriptions,
        **kwargs,
    )
