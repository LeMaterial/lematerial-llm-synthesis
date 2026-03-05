"""Pipeline for extracting synthesis procedures with performance data linking.

This pipeline combines:
1. Material extraction from paper text
2. Synthesis procedure extraction for each material
3. Figure/plot extraction from paper
4. Performance data linking (matching plot series to materials)
"""

import asyncio
import json
import logging
import os
from typing import Any

from pydantic import BaseModel, Field

from llm_synthesis.config.plot_filter_config import PlotFilterConfig
from llm_synthesis.models.figure import FigureInfo, FigureInfoWithPaper
from llm_synthesis.models.ontologies.general import GeneralSynthesisOntology
from llm_synthesis.models.paper import Paper, SynthesisEntry
from llm_synthesis.models.performance import (
    LinkingStats,
    MaterialPerformanceData,
    PlotMaterialMapping,
)
from llm_synthesis.models.plot import ExtractedLinePlotData
from llm_synthesis.transformers.performance_linking.base import LinkingInput
from llm_synthesis.transformers.performance_linking.plot_filter import (
    PlotFilter,
)
from llm_synthesis.transformers.performance_linking.series_material_linker import (  # noqa: E501
    SeriesMaterialLinker,
)
from llm_synthesis.utils import clean_text
from llm_synthesis.utils.concurrency import run_with_semaphore
from llm_synthesis.utils.figure_utils import clean_text_from_images
from llm_synthesis.utils.performance_utils import (
    aggregate_all_materials_performance,
    compute_linking_stats,
    sanitize_filename,
)

logger = logging.getLogger(__name__)


class SynthesisWithPerformanceEntry(BaseModel):
    """A material's synthesis procedure with linked performance data."""

    material: str
    synthesis: GeneralSynthesisOntology | None = None
    evaluation: Any | None = None  # GeneralSynthesisEvaluation
    performance: MaterialPerformanceData | None = None
    linking_evaluation: Any | None = None  # LinkingEvaluation


class PipelineResult(BaseModel):
    """Complete result from the synthesis + performance pipeline."""

    paper_id: str
    paper_name: str
    materials: list[str]
    results: list[SynthesisWithPerformanceEntry]
    plot_mappings: list[PlotMaterialMapping] = Field(default_factory=list)
    num_plots: int = 0
    linking_stats: LinkingStats | None = None
    materials_with_performance: list[str] = Field(default_factory=list)
    materials_without_performance: list[str] = Field(default_factory=list)


class SynthesisPerformancePipeline:
    """End-to-end pipeline: Paper → Materials → Synthesis → Performance Linking.

    This pipeline processes scientific papers to extract:
    1. Materials synthesized in the paper
    2. Detailed synthesis procedures for each material
    3. Performance data from plots, linked to specific materials

    The pipeline is modular - each component can be customized or replaced.
    """

    def __init__(
        self,
        material_extractor,
        synthesis_extractor,
        judge=None,
        linking_judge=None,
        plot_extractor=None,
        series_linker: SeriesMaterialLinker | None = None,
        plot_filter_config: PlotFilterConfig | None = None,
    ):
        """Initialize the pipeline.

        Args:
            material_extractor: Extractor for identifying materials in paper
            synthesis_extractor: Extractor for synthesis procedures
            judge: Optional judge for evaluating synthesis quality
            linking_judge: Optional judge for evaluating linking quality
            plot_extractor: Optional plot extractor
                (e.g. ClaudeLinePlotDataExtractor).
            series_linker: Optional linker for matching series to materials
            plot_filter_config: Optional config for filtering plots
        """
        self.material_extractor = material_extractor
        self.synthesis_extractor = synthesis_extractor
        self.judge = judge
        self.linking_judge = linking_judge
        self.plot_extractor = plot_extractor
        self.series_linker = series_linker
        self.plot_filter = (
            PlotFilter(plot_filter_config)
            if plot_filter_config
            else PlotFilter()
        )

    def extract_materials(self, paper_text: str) -> list[str]:
        """Step 1: Extract list of materials from paper text.

        Args:
            paper_text: Full paper text

        Returns:
            List of material names
        """
        logger.info("Step 1: Extracting materials...")
        materials_text = self.material_extractor.forward(
            input=clean_text(paper_text)
        )

        if not materials_text:
            logger.warning("  No materials found")
            return []

        materials = [
            m.strip()
            for m in materials_text.replace("\n", ",").split(",")
            if m.strip()
        ]
        logger.info(f"  Found {len(materials)} materials: {materials}")
        return materials

    def extract_synthesis(
        self,
        paper_text: str,
        material: str,
    ) -> tuple[GeneralSynthesisOntology, Any]:
        """Step 2: Extract synthesis procedure for a single material.

        Args:
            paper_text: Full paper text
            material: Material name to extract synthesis for

        Returns:
            Tuple of (synthesis ontology, evaluation result or None)
        """
        logger.info(f"Step 2: Extracting synthesis for '{material}'...")

        try:
            synthesis = self.synthesis_extractor.forward(
                input=(clean_text(paper_text), material)
            )

            # Evaluate if judge is available
            evaluation = None
            if self.judge:
                try:
                    evaluation = self.judge.forward(
                        (
                            clean_text(paper_text),
                            json.dumps(synthesis.model_dump()),
                            material,
                        )
                    )
                    logger.info(
                        f"  Evaluation score: "
                        f"{evaluation.scores.overall_score}/5.0"
                    )
                except Exception as e:
                    logger.warning(f"  Judge evaluation failed: {e}")

            return synthesis, evaluation

        except Exception as e:
            logger.error(f"  Synthesis extraction failed: {e}")
            return (
                GeneralSynthesisOntology(
                    target_compound=material,
                    target_compound_type="other",
                    synthesis_method="other",
                    notes=f"Extraction failed: {e}",
                ),
                None,
            )

    def extract_figures(self, markdown_text: str) -> list[FigureInfo]:
        """Step 3: Extract and classify figures from markdown.

        Args:
            markdown_text: Markdown text with embedded base64 images

        Returns:
            List of quantitative figure info objects
        """
        logger.info("Step 3: Extracting figures...")

        try:
            from llm_synthesis.transformers.figure_extraction.regex_figure_extractor import (  # noqa: E501
                FigureExtractorMarkdown,
            )

            extractor = FigureExtractorMarkdown()
            all_figures = extractor.forward(markdown_text)

            # Filter to only quantitative figures (classified by ResNet).
            # This avoids sending non-quantitative figures (schematics,
            # microscopy, etc.) to the Claude VLM, saving compute.
            quantitative_figures = [f for f in all_figures if f.quantitative]
            logger.info(
                f"  Found {len(all_figures)} figures, "
                f"{len(quantitative_figures)} quantitative"
            )
            return quantitative_figures

        except Exception as e:
            logger.warning(f"  Figure extraction failed: {e}")
            return []

    def _extract_one_plot(
        self,
        fig: FigureInfo,
        paper_text: str,
        si_text: str,
    ) -> tuple[ExtractedLinePlotData | None, FigureInfo]:
        """Extract plot data for a single figure (used by async path)."""
        if not self.plot_extractor:
            return None, fig
        fig_with_paper = FigureInfoWithPaper(
            base64_data=fig.base64_data,
            alt_text=fig.alt_text,
            position=fig.position,
            context_before=fig.context_before,
            context_after=fig.context_after,
            figure_reference=fig.figure_reference,
            figure_class=fig.figure_class,
            quantitative=fig.quantitative,
            paper_text=clean_text_from_images(paper_text),
            si_text=si_text,
        )
        try:
            plot_data = self.plot_extractor.forward(fig_with_paper)
            if plot_data and plot_data.name_to_coordinates:
                return plot_data, fig
        except Exception as e:
            logger.warning(
                f"    {fig.figure_reference}: extraction failed - {e}"
            )
        return None, fig

    def extract_plot_data(
        self,
        figures: list[FigureInfo],
        paper_text: str,
        si_text: str = "",
    ) -> tuple[list[ExtractedLinePlotData], list[FigureInfo]]:
        """Step 4: Extract data from quantitative plots.

        Args:
            figures: List of FigureInfo for quantitative figures
            paper_text: Full paper text for context
            si_text: Supplementary information text

        Returns:
            Tuple of (list of plot data, list of corresponding figures)
        """
        if not self.plot_extractor:
            logger.info(
                "Step 4: Skipping plot extraction (no extractor configured)"
            )
            return [], []

        logger.info(f"Step 4: Extracting data from {len(figures)} plots...")
        plots = []
        plot_figures = []

        for fig in figures:
            plot_data, _ = self._extract_one_plot(fig, paper_text, si_text)
            if plot_data is not None:
                plots.append(plot_data)
                plot_figures.append(fig)
                logger.info(
                    f"    {fig.figure_reference}: "
                    f"{len(plot_data.name_to_coordinates)} series extracted"
                )

        logger.info(f"  Extracted data from {len(plots)} plots")
        return plots, plot_figures

    def _link_one_plot(
        self,
        idx: int,
        plot: ExtractedLinePlotData,
        fig: FigureInfo,
        materials: list[str],
    ) -> PlotMaterialMapping | None:
        """Link one plot to materials (used by sync and async path)."""
        if not self.series_linker:
            return None
        series_names = list(plot.name_to_coordinates.keys())
        if not series_names:
            return None
        context = f"{fig.context_before} {fig.context_after}"
        plot_meta = {
            "title": plot.title,
            "x_axis_label": plot.x_axis_label,
            "x_axis_unit": plot.x_axis_unit,
            "y_left_axis_label": plot.y_left_axis_label,
            "y_left_axis_unit": plot.y_left_axis_unit,
        }
        linking_input = LinkingInput(
            materials=materials,
            series_names=series_names,
            context=context,
            plot_metadata=plot_meta,
        )
        validated_mappings = self.series_linker.forward(linking_input)
        matched_series = {m.series_name for m in validated_mappings}
        unmatched = [s for s in series_names if s not in matched_series]
        return PlotMaterialMapping(
            plot_index=idx,
            figure_reference=fig.figure_reference,
            mappings=validated_mappings,
            unmatched_series=unmatched,
        )

    def link_performance(
        self,
        materials: list[str],
        plots: list[ExtractedLinePlotData],
        figures: list[FigureInfo],
    ) -> tuple[list[PlotMaterialMapping], LinkingStats]:
        """Step 5: Link plot series to materials.

        Args:
            materials: List of material names
            plots: List of extracted plot data
            figures: List of corresponding figure info

        Returns:
            Tuple of (list of mappings, linking statistics)
        """
        if not self.series_linker:
            logger.info(
                "Step 5: Skipping performance linking (no linker configured)"
            )
            return [], LinkingStats(total_plots_extracted=len(plots))

        logger.info(
            f"Step 5: Linking {len(plots)} plots to {len(materials)} "
            "materials..."
        )

        # Filter plots
        relevant_plots, skip_counts = self.plot_filter.filter_plots(plots)
        skipped_plots = []  # Could be enhanced to track details

        all_mappings = []
        for idx, plot in relevant_plots:
            fig = figures[idx]
            mapping = self._link_one_plot(idx, plot, fig, materials)
            if mapping is not None:
                all_mappings.append(mapping)
                logger.info(
                    f"    Linking plot {idx} '{plot.title or 'N/A'}' "
                    f"({len(plot.name_to_coordinates)} series)"
                )
                logger.info(
                    f"      Matched: {len(mapping.mappings)}, "
                    f"Unmatched: {mapping.unmatched_series}"
                )

        # Compute stats
        stats = compute_linking_stats(
            total_plots=len(plots),
            mappings=all_mappings,
            skip_counts=skip_counts,
            skipped_plots=skipped_plots,
        )

        return all_mappings, stats

    def _evaluate_linking(
        self,
        paper_text: str,
        all_syntheses: list[SynthesisEntry],
        plots: list[ExtractedLinePlotData],
        plot_mappings: list[PlotMaterialMapping],
        performance_data: dict[str, MaterialPerformanceData],
    ) -> Any | None:
        """Step 6: Evaluate linking quality with the linking judge.

        Calls the linking judge once per paper with the full context:
        paper text, all extracted syntheses, all plot data, and the
        linking output.

        Args:
            paper_text: Full paper text
            all_syntheses: All synthesis entries for the paper
            plots: All extracted plot data
            plot_mappings: All plot-to-material mappings (linking output)
            performance_data: Aggregated performance per material

        Returns:
            LinkingEvaluation or None if judge fails
        """
        logger.info("Step 6: Evaluating linking quality...")
        try:
            synthesis_json = json.dumps(
                [
                    {
                        "material": e.material,
                        "synthesis": e.synthesis.model_dump()
                        if e.synthesis
                        else None,
                    }
                    for e in all_syntheses
                ],
                indent=2,
            )
            plot_data_json = json.dumps(
                [p.model_dump() for p in plots],
                indent=2,
            )
            linking_output_json = json.dumps(
                {
                    "mappings": [m.model_dump() for m in plot_mappings],
                    "performance_per_material": {
                        k: v.model_dump() for k, v in performance_data.items()
                    },
                },
                indent=2,
            )

            evaluation = self.linking_judge.forward(
                (
                    clean_text(paper_text),
                    synthesis_json,
                    plot_data_json,
                    linking_output_json,
                )
            )
            logger.info(
                f"  Linking evaluation score: "
                f"{evaluation.scores.overall_score}/5.0"
            )
            if evaluation.failure_flags.active_flags():
                logger.info(
                    f"  Failure flags: "
                    f"{evaluation.failure_flags.active_flags()}"
                )
            return evaluation

        except Exception as e:
            logger.warning(f"  Linking judge evaluation failed: {e}")
            return None

    def process_paper(
        self,
        paper: Paper,
        skip_figures: bool = False,
    ) -> PipelineResult | None:
        """Process a single paper through the full pipeline.

        Args:
            paper: Paper object with text content
            skip_figures: If True, skip figures and performance linking

        Returns:
            PipelineResult or None if processing failed
        """
        logger.info(f"Processing: {paper.name}")

        # Step 1: Extract materials
        materials = self.extract_materials(paper.publication_text)
        if not materials:
            logger.warning("  No materials found, skipping paper")
            return None

        # Step 2: Extract synthesis for each material
        all_syntheses = []
        for material in materials:
            synthesis, evaluation = self.extract_synthesis(
                paper.publication_text, material
            )
            all_syntheses.append(
                SynthesisEntry(
                    material=material,
                    synthesis=synthesis,
                    evaluation=evaluation,
                )
            )

        # Steps 3-5: Figure/plot extraction and linking (optional)
        performance_data = {}
        plot_mappings = []
        extracted_plots = []
        linking_stats = None
        linking_evaluation = None

        if not skip_figures:
            # Step 3: Extract figures
            figures = self.extract_figures(paper.publication_text)

            if figures:
                # Step 4: Extract plot data
                plots, plot_figures = self.extract_plot_data(
                    figures, paper.publication_text, paper.si_text
                )
                extracted_plots = plots

                if plots:
                    # Step 5: Link performance (with graceful error handling)
                    try:
                        plot_mappings, linking_stats = self.link_performance(
                            materials, plots, plot_figures
                        )

                        # Aggregate per material
                        performance_data = aggregate_all_materials_performance(
                            materials, plot_mappings, plots
                        )

                        # Step 6: Evaluate linking quality (optional)
                        if self.linking_judge and plot_mappings:
                            linking_evaluation = self._evaluate_linking(
                                paper_text=paper.publication_text,
                                all_syntheses=all_syntheses,
                                plots=plots,
                                plot_mappings=plot_mappings,
                                performance_data=performance_data,
                            )
                    except Exception as e:
                        logger.warning(
                            f"  Performance linking failed: {e}. "
                            "Synthesis results saved without performance data."
                        )
                        # Keep empty defaults - synthesis still saved

        # Build results
        results = []
        for entry in all_syntheses:
            results.append(
                SynthesisWithPerformanceEntry(
                    material=entry.material,
                    synthesis=entry.synthesis,
                    evaluation=entry.evaluation,
                    performance=performance_data.get(entry.material),
                    linking_evaluation=linking_evaluation,
                )
            )

        # Summary
        materials_with_perf = [m for m in materials if m in performance_data]
        materials_without_perf = [
            m for m in materials if m not in performance_data
        ]

        return PipelineResult(
            paper_id=paper.id,
            paper_name=paper.name,
            materials=materials,
            results=results,
            plot_mappings=plot_mappings,
            num_plots=len(extracted_plots),
            linking_stats=linking_stats,
            materials_with_performance=materials_with_perf,
            materials_without_performance=materials_without_perf,
        )

    async def process_paper_async(
        self,
        paper: Paper,
        semaphore: asyncio.Semaphore,
        skip_figures: bool = False,
    ) -> PipelineResult | None:
        """Process one paper with concurrent LLM calls (asyncio + semaphore).

        Same as process_paper but runs independent LLM calls in parallel:
        - Materials: one call, then synthesis+judge per material in parallel
        - Plot extraction: one call per figure in parallel
        - Linking: one call per plot in parallel

        Args:
            paper: Paper object with text content
            semaphore: Cap on concurrent LLM calls
            skip_figures: If True, skip figures and performance linking

        Returns:
            PipelineResult or None if processing failed
        """
        logger.info(f"Processing: {paper.name}")

        # Step 1: Material extraction (one call)
        materials_text = await run_with_semaphore(
            semaphore,
            self.material_extractor.forward,
            input=clean_text(paper.publication_text),
        )
        if not materials_text:
            logger.warning("  No materials found")
            return None
        materials = [
            m.strip()
            for m in materials_text.replace("\n", ",").split(",")
            if m.strip()
        ]
        logger.info(f"  Found {len(materials)} materials: {materials}")

        # Step 2: Synthesis + judge per material (parallel)
        async def extract_synthesis_one(material: str) -> SynthesisEntry:
            try:
                synthesis = await run_with_semaphore(
                    semaphore,
                    self.synthesis_extractor.forward,
                    input=(clean_text(paper.publication_text), material),
                )
                evaluation = None
                if self.judge:
                    try:
                        evaluation = await run_with_semaphore(
                            semaphore,
                            self.judge.forward,
                            (
                                clean_text(paper.publication_text),
                                json.dumps(synthesis.model_dump()),
                                material,
                            ),
                        )
                        logger.info(
                            f"  [{material}] Evaluation score: "
                            f"{evaluation.scores.overall_score}/5.0"
                        )
                    except Exception as e:
                        logger.warning(f"  Judge evaluation failed: {e}")
                return SynthesisEntry(
                    material=material,
                    synthesis=synthesis,
                    evaluation=evaluation,
                )
            except Exception as e:
                logger.error(
                    f"  Synthesis extraction failed for {material}: {e}"
                )
                return SynthesisEntry(
                    material=material,
                    synthesis=GeneralSynthesisOntology(
                        target_compound=material,
                        target_compound_type="other",
                        synthesis_method="other",
                        notes=f"Extraction failed: {e}",
                    ),
                    evaluation=None,
                )

        all_syntheses = await asyncio.gather(
            *[extract_synthesis_one(mat) for mat in materials]
        )
        all_syntheses = list(all_syntheses)

        # Steps 3-5: Figures, plot extraction, linking (optional)
        performance_data = {}
        plot_mappings = []
        extracted_plots = []
        linking_stats = None
        linking_evaluation = None

        if not skip_figures:
            # Step 3: Extract figures (CPU-bound, no LLM — run in thread directly)
            figures = await asyncio.to_thread(
                self.extract_figures, paper.publication_text
            )

            if figures:
                # Step 4: Extract plot data in parallel (one call per figure)
                paper_text = paper.publication_text
                si_text = paper.si_text or ""
                plot_results = await asyncio.gather(
                    *[
                        run_with_semaphore(
                            semaphore,
                            self._extract_one_plot,
                            fig,
                            paper_text,
                            si_text,
                        )
                        for fig in figures
                    ]
                )
                plots = []
                plot_figures = []
                for plot_data, fig in plot_results:
                    if plot_data is not None:
                        plots.append(plot_data)
                        plot_figures.append(fig)
                extracted_plots = plots
                logger.info(f"  Extracted data from {len(plots)} plots")

                if plots:
                    try:
                        # Step 5: Link each plot in parallel
                        relevant_plots, skip_counts = (
                            self.plot_filter.filter_plots(plots)
                        )
                        skipped_plots = []
                        link_tasks = [
                            run_with_semaphore(
                                semaphore,
                                self._link_one_plot,
                                idx,
                                plot,
                                plot_figures[idx],
                                materials,
                            )
                            for idx, plot in relevant_plots
                        ]
                        mapping_results = await asyncio.gather(*link_tasks)
                        all_mappings = [
                            m for m in mapping_results if m is not None
                        ]
                        plot_mappings = all_mappings
                        linking_stats = compute_linking_stats(
                            total_plots=len(plots),
                            mappings=all_mappings,
                            skip_counts=skip_counts,
                            skipped_plots=skipped_plots,
                        )
                        performance_data = aggregate_all_materials_performance(
                            materials, plot_mappings, plots
                        )

                        # Step 6: Linking judge (one call)
                        if self.linking_judge and plot_mappings:
                            synthesis_json = json.dumps(
                                [
                                    {
                                        "material": e.material,
                                        "synthesis": (
                                            e.synthesis.model_dump()
                                            if e.synthesis
                                            else None
                                        ),
                                    }
                                    for e in all_syntheses
                                ],
                                indent=2,
                            )
                            plot_data_json = json.dumps(
                                [p.model_dump() for p in plots], indent=2
                            )
                            linking_output_json = json.dumps(
                                {
                                    "mappings": [
                                        m.model_dump() for m in plot_mappings
                                    ],
                                    "performance_per_material": {
                                        k: v.model_dump()
                                        for k, v in performance_data.items()
                                    },
                                },
                                indent=2,
                            )
                            linking_evaluation = await run_with_semaphore(
                                semaphore,
                                self.linking_judge.forward,
                                (
                                    clean_text(paper_text),
                                    synthesis_json,
                                    plot_data_json,
                                    linking_output_json,
                                ),
                            )
                            if linking_evaluation:
                                logger.info(
                                    f"  Linking evaluation score: "
                                    f"{linking_evaluation.scores.overall_score}/5.0"
                                )
                    except Exception as e:
                        logger.warning(
                            f"  Performance linking failed: {e}. "
                            "Synthesis saved without performance."
                        )

        # Build results
        results = [
            SynthesisWithPerformanceEntry(
                material=entry.material,
                synthesis=entry.synthesis,
                evaluation=entry.evaluation,
                performance=performance_data.get(entry.material),
                linking_evaluation=linking_evaluation,
            )
            for entry in all_syntheses
        ]
        materials_with_perf = [m for m in materials if m in performance_data]
        materials_without_perf = [
            m for m in materials if m not in performance_data
        ]

        return PipelineResult(
            paper_id=paper.id,
            paper_name=paper.name,
            materials=materials,
            results=results,
            plot_mappings=plot_mappings,
            num_plots=len(extracted_plots),
            linking_stats=linking_stats,
            materials_with_performance=materials_with_perf,
            materials_without_performance=materials_without_perf,
        )

    def save_results(self, result: PipelineResult, output_dir: str) -> None:
        """Save pipeline results to disk.

        Args:
            result: PipelineResult to save
            output_dir: Base output directory
        """
        paper_dir = os.path.join(output_dir, result.paper_id)
        os.makedirs(paper_dir, exist_ok=True)

        # One file per material
        for entry in result.results:
            mat_name = sanitize_filename(entry.material)
            mat_path = os.path.join(paper_dir, f"{mat_name}.json")
            with open(mat_path, "w") as f:
                json.dump(entry.model_dump(), f, indent=2)

        # Plot mappings
        mappings_path = os.path.join(paper_dir, "performance_mappings.json")
        with open(mappings_path, "w") as f:
            json.dump(
                [m.model_dump() for m in result.plot_mappings], f, indent=2
            )

        # Summary
        summary = {
            "paper_id": result.paper_id,
            "paper_name": result.paper_name,
            "total_materials": len(result.materials),
            "materials_with_performance": len(
                result.materials_with_performance
            ),
            "materials_without_performance": len(
                result.materials_without_performance
            ),
            "materials_list": result.materials,
            "materials_with_performance_list": (
                result.materials_with_performance
            ),
            "materials_without_performance_list": (
                result.materials_without_performance
            ),
            "total_plots_extracted": result.num_plots,
        }

        if result.linking_stats:
            stats = result.linking_stats
            summary["plots_linked"] = stats.plots_linked
            summary["plots_skipped"] = {
                "not_relevant_x": stats.plots_skipped_not_relevant_x,
                "not_relevant_y": stats.plots_skipped_not_relevant_y,
                "no_series": stats.plots_skipped_no_series,
            }
            summary["confidence_breakdown"] = stats.confidence_counts
            summary["all_unmatched_series"] = stats.all_unmatched_series

        summary_path = os.path.join(paper_dir, "linking_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(
            f"  Saved {len(result.results)} material files to {paper_dir}/"
        )
