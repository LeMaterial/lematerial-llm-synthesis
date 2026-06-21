"""Shared batch-processing runner for all case studies.

BatchRunner wires together:
  - SynthesisPerformancePipeline (material + synthesis + figure + linking)
  - Optional domain text-metric extraction (BaseTextMetricExtractor)
  - Optional domain VLM post-processing (BaseVLMMetricProcessor)
  - An output writer (BaseOutputWriter)

Case study scripts instantiate BatchRunner with a DomainConfig and call
runner.run().  All SI detection, loop control, rate-limit handling, and
component initialisation live here — not in the case study scripts.

Usage::

    from llm_synthesis.config.domain_config import DomainConfig
    from llm_synthesis.runners.batch_runner import BatchRunner

    runner = BatchRunner(
        domain_config=DomainConfig.for_catalysis(),
        gemini_model="gemini-3.0-flash",
        claude_model="claude-sonnet-4-20250514",
    )
    runner.run(pdf_dir="/path/to/pdfs", output_dir="/path/to/results")
"""

import logging
import time
from pathlib import Path

from llm_synthesis.config.domain_config import DomainConfig
from llm_synthesis.config.synthesis_prompts import SYNTHESIS_SYSTEM_PROMPT
from llm_synthesis.metrics.judge.general_synthesis_judge import (
    DspyGeneralSynthesisJudge,
    make_general_synthesis_judge_signature,
)
from llm_synthesis.metrics.judge.linking_judge import (
    DspyLinkingJudge,
    make_linking_judge_signature,
)
from llm_synthesis.models.paper import Paper
from llm_synthesis.services.pipelines.synthesis_performance_pipeline import (
    SynthesisPerformancePipeline,
)
from llm_synthesis.transformers.material_extraction.dspy_extraction import (
    DspyTextExtractor,
    make_dspy_text_extractor_signature,
)
from llm_synthesis.transformers.pdf_extraction import MistralPDFExtractor
from llm_synthesis.transformers.performance_linking.series_material_linker import (  # noqa: E501
    SeriesMaterialLinker,
)
from llm_synthesis.transformers.plot_extraction.claude_extraction.plot_data_extraction import (  # noqa: E501
    ClaudeLinePlotDataExtractor,
)
from llm_synthesis.transformers.plot_extraction.litellm_plot_data_extraction import (  # noqa: E501
    LiteLLMPlotDataExtractor,
)
from llm_synthesis.transformers.synthesis_extraction.dspy_synthesis_extraction import (  # noqa: E501
    DspySynthesisExtractor,
    make_dspy_synthesis_extractor_signature,
)
from llm_synthesis.utils.dspy_utils import get_llm_from_name
from llm_synthesis.utils.llms import LLM_REGISTRY
from llm_synthesis.utils.si_utils import (
    find_si_file,
    is_si_file,
    load_file_text,
)

logger = logging.getLogger(__name__)


class BatchRunner:
    """Run the full extraction pipeline on every PDF in a folder.

    Args:
        domain_config: Domain-specific configuration (plot filter, material
            instructions, optional metric extractors, output writer).
        gemini_model: Gemini model for material/synthesis extraction and judges.
        claude_model: Claude model for plot data extraction and (optionally)
            VLM metric processing.
        linker_model: Gemini model for series-to-material linking.
        material_model: Override model for material extraction only.
            Defaults to ``gemini_model`` if not set.
        synthesis_max_tokens: Max tokens for synthesis LLM call.
        linker_max_tokens: Max tokens for the series linker LLM call.
    """

    def __init__(
        self,
        domain_config: DomainConfig,
        gemini_model: str = "gemini-3.0-flash",
        claude_model: str = "claude-sonnet-4-20250514",
        linker_model: str = "gemini-3.0-flash",
        material_model: str | None = None,
        synthesis_max_tokens: int = 80_000,
        linker_max_tokens: int = 32_000,
        plot_vlm: str | None = None,
    ) -> None:
        self.domain_config = domain_config
        self.gemini_model = gemini_model
        self.claude_model = claude_model
        self.linker_model = linker_model
        self.material_model = material_model or gemini_model
        self.synthesis_max_tokens = synthesis_max_tokens
        self.linker_max_tokens = linker_max_tokens
        # plot_vlm: LLM_REGISTRY key (e.g. "gemini-3-flash") or raw litellm
        # model string. When set, overrides claude_model for plot extraction
        # using LiteLLMPlotDataExtractor instead of ClaudeLinePlotDataExtractor.
        self.plot_vlm = plot_vlm

        self._pdf_extractor: MistralPDFExtractor | None = None
        self._pipeline: SynthesisPerformancePipeline | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        pdf_dir: str | Path,
        output_dir: str | Path,
        max_papers: int | None = None,
        skip_existing: bool = False,
        skip_figures: bool = False,
        phase: str = "all",
        cache_dir: str | Path | None = None,
    ) -> None:
        """Process every PDF in pdf_dir and write results to output_dir.

        Args:
            pdf_dir: Directory containing PDF (and optionally MD/TXT) files.
            output_dir: Root directory for output files.
            max_papers: If set, stop after this many papers (useful for
                testing).
            skip_existing: If True, skip papers whose output already exists.
            skip_figures: If True, skip figure/plot extraction entirely.
            phase: One of:
                "all"       — full pipeline (default, original behaviour)
                "synthesis" — OCR + materials + synthesis + figure detection
                              only; saves cache to cache_dir/_cache/<paper_id>/
                "vlm"       — loads cached synthesis+figures, runs VLM plot
                              extraction + linking + writes output_dir
            cache_dir: Root dir for synthesis/figure cache.  Defaults to
                output_dir when phase="synthesis", and must be provided when
                phase="vlm".
        """
        import warnings

        warnings.filterwarnings(
            "ignore", category=UserWarning, module="pydantic"
        )
        logging.getLogger("LiteLLM").setLevel(logging.ERROR)
        logging.getLogger("litellm").setLevel(logging.ERROR)
        logging.getLogger("pydantic").setLevel(logging.ERROR)

        if phase not in ("all", "synthesis", "vlm"):
            raise ValueError(
                f"phase must be 'all', 'synthesis', or 'vlm', got {phase!r}"
            )

        pdf_dir = Path(pdf_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Resolve cache_dir
        _cache_dir: Path
        if cache_dir is not None:
            _cache_dir = Path(cache_dir)
        elif phase == "vlm":
            raise ValueError("cache_dir is required when phase='vlm'")
        else:
            _cache_dir = output_dir

        if phase == "vlm":
            # VLM-only: no PDFs needed, iterate over cached paper dirs
            self._init_components()
            cache_root = _cache_dir / "_cache"
            if not cache_root.exists():
                raise FileNotFoundError(f"Cache not found: {cache_root}")
            paper_dirs = sorted(d for d in cache_root.iterdir() if d.is_dir())
            if max_papers:
                paper_dirs = paper_dirs[:max_papers]
            all_summaries: list[dict] = []
            total_start = time.time()
            for i, paper_cache_dir in enumerate(paper_dirs, 1):
                logger.info(
                    "\n%s\n# VLM PHASE %d/%d: %s\n%s",
                    "#" * 70,
                    i,
                    len(paper_dirs),
                    paper_cache_dir.name,
                    "#" * 70,
                )
                try:
                    summary = self._phase2_vlm(paper_cache_dir, output_dir)
                    all_summaries.append(summary)
                except Exception as e:
                    logger.error("FAILED %s: %s", paper_cache_dir.name, e)
                    import traceback

                    traceback.print_exc()
                    all_summaries.append(
                        {"paper_id": paper_cache_dir.name, "error": str(e)}
                    )
            total_elapsed = round(time.time() - total_start, 1)
            self.domain_config.output_writer.finalize(output_dir, all_summaries)
            self._print_batch_summary(all_summaries, total_elapsed, output_dir)
            return

        if not pdf_dir.is_dir():
            raise ValueError(f"Not a directory: {pdf_dir}")

        papers = self._discover_papers(pdf_dir)

        if skip_existing:
            papers = self._filter_existing(papers, output_dir)

        if max_papers and len(papers) > max_papers:
            logger.info("Limiting to first %d papers (--max flag)", max_papers)
            papers = papers[:max_papers]

        logger.info(
            "Domain: %s | Papers to process: %d | Output: %s",
            self.domain_config.name,
            len(papers),
            output_dir,
        )

        if not papers:
            logger.warning("No papers to process.")
            return

        self._init_components()

        all_summaries: list[dict] = []
        total_start = time.time()

        for i, paper_path in enumerate(papers, 1):
            logger.info(
                "\n%s\n# PAPER %d/%d: %s\n%s",
                "#" * 70,
                i,
                len(papers),
                paper_path.name,
                "#" * 70,
            )
            try:
                if phase == "synthesis":
                    summary = self._phase1_synthesis(paper_path, _cache_dir)
                else:
                    summary = self._process_one(
                        paper_path, output_dir, skip_figures=skip_figures
                    )
                all_summaries.append(summary)
            except Exception as e:
                error_str = str(e).lower()
                if (
                    ("rate" in error_str and "limit" in error_str)
                    or "429" in error_str
                    or "quota" in error_str
                    or "resource_exhausted" in error_str
                ):
                    logger.error("RATE LIMIT — stopping batch: %s", e)
                    all_summaries.append(
                        {
                            "paper_id": paper_path.stem,
                            "error": f"RATE_LIMIT: {e}",
                        }
                    )
                    break
                logger.error("FAILED %s: %s", paper_path.name, e)
                traceback.print_exc()
                all_summaries.append(
                    {"paper_id": paper_path.stem, "error": str(e)}
                )

        total_elapsed = round(time.time() - total_start, 1)

        # Attach total timing to the summary list before finalize
        for s in all_summaries:
            if "total_time_seconds" not in s:
                pass  # per-paper timing already in each summary
        self.domain_config.output_writer.finalize(output_dir, all_summaries)

        self._print_batch_summary(all_summaries, total_elapsed, output_dir)

    # ------------------------------------------------------------------
    # Component initialisation
    # ------------------------------------------------------------------

    def _init_components(self) -> None:
        """Build PDF extractor, pipeline, and (optionally) text metric
        extractor."""
        logger.info(
            "Initialising pipeline for domain '%s'...", self.domain_config.name
        )

        self._pdf_extractor = MistralPDFExtractor(structured=False)

        cfg = self.domain_config

        # Material extractor
        material_sig = make_dspy_text_extractor_signature(
            instructions=cfg.material_extraction_instructions,
            output_description=cfg.material_output_description,
        )
        material_lm = get_llm_from_name(
            self.material_model,
            model_kwargs={"temperature": 0.0, "max_tokens": 16_000},
        )
        material_extractor = DspyTextExtractor(
            signature=material_sig, lm=material_lm
        )

        # Synthesis extractor
        synthesis_sig = make_dspy_synthesis_extractor_signature(
            instructions=(
                "Extract the complete structured synthesis procedure for the "
                "specified material. Include all steps, conditions "
                "(temperature, "
                "time, atmosphere), equipment, and precursors. "
                "Be thorough and preserve all quantitative details."
            ),
        )
        synthesis_lm = get_llm_from_name(
            self.gemini_model,
            model_kwargs={
                "temperature": 0.0,
                "max_tokens": self.synthesis_max_tokens,
                "num_retries": 3,
            },
            system_prompt=SYNTHESIS_SYSTEM_PROMPT,
        )
        synthesis_extractor = DspySynthesisExtractor(
            signature=synthesis_sig, lm=synthesis_lm
        )

        # Synthesis judge
        judge_lm = get_llm_from_name(
            self.gemini_model,
            model_kwargs={"temperature": 0.1, "max_tokens": 20_000},
        )
        judge = DspyGeneralSynthesisJudge(
            signature=make_general_synthesis_judge_signature(), lm=judge_lm
        )

        # Plot extractor — LiteLLM (any registry VLM) or Claude direct
        if self.plot_vlm is not None:
            if self.plot_vlm in LLM_REGISTRY.configs:
                cfg_vlm = LLM_REGISTRY.configs[self.plot_vlm]
                plot_extractor = LiteLLMPlotDataExtractor(
                    model=cfg_vlm.model,
                    api_key=cfg_vlm.api_key,
                    api_base=cfg_vlm.api_base,
                    extra_kwargs=cfg_vlm.extra_kwargs or {},
                )
            else:
                plot_extractor = LiteLLMPlotDataExtractor(model=self.plot_vlm)
        else:
            plot_extractor = ClaudeLinePlotDataExtractor(
                model_name=self.claude_model
            )

        # Series linker
        linker_lm = get_llm_from_name(
            self.linker_model,
            model_kwargs={
                "temperature": 0.0,
                "max_tokens": self.linker_max_tokens,
            },
        )
        series_linker = SeriesMaterialLinker(lm=linker_lm)

        # Linking judge
        linking_judge_lm = get_llm_from_name(
            self.gemini_model,
            model_kwargs={"temperature": 0.1, "max_tokens": 60_000},
        )
        linking_judge = DspyLinkingJudge(
            signature=make_linking_judge_signature(), lm=linking_judge_lm
        )

        self._pipeline = SynthesisPerformancePipeline(
            material_extractor=material_extractor,
            synthesis_extractor=synthesis_extractor,
            judge=judge,
            linking_judge=linking_judge,
            plot_extractor=plot_extractor,
            series_linker=series_linker,
            plot_filter_config=cfg.plot_filter_config,
        )

        # Wire up text metric extractor for superconductors domain:
        # DomainConfig.for_superconductivity() sets text_metric_extractor=None
        # as a sentinel so we can inject the gemini_model here.
        if cfg.text_metric_extractor is None and cfg.name == "superconductors":
            from llm_synthesis.domain_metrics.superconductors.tc_text_extractor import (  # noqa: E501
                TcTextExtractor,
            )

            tc_lm = get_llm_from_name(
                self.gemini_model,
                model_kwargs={"temperature": 0.0, "max_tokens": 16_384},
            )
            # Mutate a copy so we don't modify the shared DomainConfig object
            import dataclasses

            self._domain_config_with_extractor = dataclasses.replace(
                cfg, text_metric_extractor=TcTextExtractor(lm=tc_lm)
            )
        else:
            self._domain_config_with_extractor = cfg

        logger.info("Pipeline ready.")

    # ------------------------------------------------------------------
    # Per-paper processing
    # ------------------------------------------------------------------

    def _process_one(
        self,
        pdf_path: Path,
        output_dir: Path,
        skip_figures: bool,
    ) -> dict:
        paper_start = time.time()
        paper_id = pdf_path.stem

        logger.info("Step 0: Loading text...")
        paper_text = load_file_text(pdf_path, self._pdf_extractor)
        logger.info("  Main paper: %d chars", len(paper_text))

        si_text = ""
        si_path = find_si_file(pdf_path)
        if si_path:
            logger.info("  Found SI: %s", si_path.name)
            try:
                si_text = load_file_text(si_path, self._pdf_extractor)
                logger.info("  SI: %d chars", len(si_text))
            except Exception as e:
                logger.warning("  SI load failed: %s", e)

        paper = Paper(
            name=paper_id,
            id=paper_id,
            publication_text=paper_text,
            si_text=si_text,
        )

        assert self._pipeline is not None
        result = self._pipeline.process_paper(paper, skip_figures=skip_figures)
        if result is None:
            raise ValueError(
                "Pipeline returned no results (no materials found?)"
            )

        cfg = self._domain_config_with_extractor

        # Optional text metric extraction
        text_metrics: dict = {}
        if cfg.text_metric_extractor is not None:
            logger.info("Running text metric extraction...")
            try:
                from llm_synthesis.utils.markdown_utils import clean_text

                text_metrics = cfg.text_metric_extractor.extract(
                    paper_text=clean_text(paper.publication_text),
                    materials=result.materials,
                )
                logger.info(
                    "  Text metrics for %d/%d materials",
                    len(text_metrics),
                    len(result.materials),
                )
            except Exception as e:
                logger.warning("Text metric extraction failed: %s", e)

        # Optional VLM metric processing
        vlm_metrics: dict = {}
        if cfg.vlm_metric_processor is not None and not skip_figures:
            if result.relevant_plots and result.plot_figures:
                logger.info("Running VLM metric processing...")
                try:
                    vlm_metrics = cfg.vlm_metric_processor.process(
                        relevant_plots=result.relevant_plots,
                        plot_figures=result.plot_figures,
                        plot_mappings=result.plot_mappings,
                        materials=result.materials,
                        paper_text=paper.publication_text,
                    )
                    logger.info(
                        "  VLM metrics for %d/%d materials",
                        len(vlm_metrics),
                        len(result.materials),
                    )
                except Exception as e:
                    logger.warning("VLM metric processing failed: %s", e)

        processing_time = time.time() - paper_start

        summary = self.domain_config.output_writer.write_paper(
            paper_id=paper_id,
            output_dir=output_dir,
            pipeline_result=result,
            text_metrics=text_metrics,
            vlm_metrics=vlm_metrics,
            processing_time=processing_time,
        )
        logger.info(
            "  Done: %d materials, %d plots, %.1fs",
            len(result.materials),
            result.num_plots,
            processing_time,
        )
        return summary

    # ------------------------------------------------------------------
    # Two-phase helpers
    # ------------------------------------------------------------------

    def _phase1_synthesis(self, pdf_path: Path, cache_root: Path) -> dict:
        """Phase 1: OCR + materials + synthesis + figure detection.

        Saves results to cache_root/_cache/<paper_id>/synthesis.json and
        figures.json so that _phase2_vlm can run cheaply for any VLM later.
        """
        import json as _json

        paper_start = time.time()
        paper_id = pdf_path.stem
        cache_dir = cache_root / "_cache" / paper_id
        cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Phase 1: Loading text for %s...", paper_id)
        paper_text = load_file_text(pdf_path, self._pdf_extractor)

        si_text = ""
        si_path = find_si_file(pdf_path)
        if si_path:
            try:
                si_text = load_file_text(si_path, self._pdf_extractor)
            except Exception as e:
                logger.warning("  SI load failed: %s", e)

        # paper = _Paper(
        #     name=paper_id,
        #     id=paper_id,
        #     publication_text=paper_text,
        #     si_text=si_text,
        # )

        assert self._pipeline is not None

        # Steps 1-2: materials + synthesis (skip_figures=True)
        materials = self._pipeline.extract_materials(paper_text)
        if not materials:
            raise ValueError("No materials found")

        from llm_synthesis.models.paper import (
            SynthesisEntry as _SE,  # noqa: N814
        )

        all_syntheses = []
        for mat in materials:
            synthesis, evaluation = self._pipeline.extract_synthesis(
                paper_text, mat
            )
            all_syntheses.append(
                _SE(material=mat, synthesis=synthesis, evaluation=evaluation)
            )

        # Step 3: figure detection only (no VLM)
        figures = self._pipeline.extract_figures(paper_text)

        # Persist synthesis
        synthesis_cache = {
            "paper_id": paper_id,
            "si_text": si_text,
            "paper_text": paper_text,
            "materials": materials,
            "syntheses": [
                {
                    "material": e.material,
                    "synthesis": e.synthesis.model_dump()
                    if e.synthesis
                    else None,
                    "evaluation": e.evaluation.model_dump()
                    if e.evaluation and hasattr(e.evaluation, "model_dump")
                    else None,
                }
                for e in all_syntheses
            ],
        }
        (cache_dir / "synthesis.json").write_text(
            _json.dumps(synthesis_cache, indent=2)
        )

        # Persist figures (base64 is a plain string — serializes as-is)
        figures_cache = [f.model_dump() for f in figures]
        (cache_dir / "figures.json").write_text(
            _json.dumps(figures_cache, indent=2)
        )

        processing_time = round(time.time() - paper_start, 1)
        logger.info(
            "  Phase 1 done: %d materials, %d figures cached, %.1fs",
            len(materials),
            len(figures),
            processing_time,
        )
        return {
            "paper_id": paper_id,
            "total_materials": len(materials),
            "figures_cached": len(figures),
            "processing_time_seconds": processing_time,
            "cache_dir": str(cache_dir),
        }

    def _phase2_vlm(self, paper_cache_dir: Path, output_dir: Path) -> dict:
        """Phase 2: Load cached synthesis+figures, run VLM + linking.

        Reads cache_dir/synthesis.json and figures.json written by
        _phase1_synthesis, runs steps 4-6, and writes normal output.
        """
        import json as _json

        paper_start = time.time()
        paper_id = paper_cache_dir.name

        syn_path = paper_cache_dir / "synthesis.json"
        fig_path = paper_cache_dir / "figures.json"
        if not syn_path.exists() or not fig_path.exists():
            raise FileNotFoundError(
                f"Cache incl for {paper_id}: need synthesis.json + figures.json"
            )

        syn_cache = _json.loads(syn_path.read_text())
        fig_cache = _json.loads(fig_path.read_text())

        from llm_synthesis.models.figure import FigureInfo as _FigureInfo
        from llm_synthesis.models.ontologies.general import (
            GeneralSynthesisOntology as _GSO,  # noqa: N814
        )
        from llm_synthesis.models.paper import (
            SynthesisEntry as _SE,  # noqa: N814
        )

        materials = syn_cache["materials"]
        paper_text = syn_cache["paper_text"]
        si_text = syn_cache.get("si_text", "")
        figures = [_FigureInfo(**f) for f in fig_cache]

        # Reconstruct syntheses
        all_syntheses = []
        for s in syn_cache["syntheses"]:
            syn_obj = _GSO(**s["synthesis"]) if s["synthesis"] else None
            all_syntheses.append(
                _SE(material=s["material"], synthesis=syn_obj, evaluation=None)
            )

        assert self._pipeline is not None

        # Step 4: VLM plot extraction
        plots, plot_figures = self._pipeline.extract_plot_data(
            figures, paper_text, si_text
        )

        # Step 5: linking
        plot_mappings: list = []
        linking_stats = None
        performance_data: dict = {}
        linking_evaluation = None
        if plots:
            from llm_synthesis.utils.performance_utils import (
                aggregate_all_materials_performance,
            )

            try:
                plot_mappings, linking_stats = self._pipeline.link_performance(
                    materials, plots, plot_figures
                )
                performance_data = aggregate_all_materials_performance(
                    materials, plot_mappings, plots
                )
                if self._pipeline.linking_judge and plot_mappings:
                    linking_evaluation = self._pipeline._evaluate_linking(
                        paper_text=paper_text,
                        all_syntheses=all_syntheses,
                        plots=plots,
                        plot_mappings=plot_mappings,
                        performance_data=performance_data,
                    )
            except Exception as e:
                logger.warning("Performance linking failed: %s", e)

        # Build PipelineResult and write output
        from llm_synthesis.services.pipelines.synthesis_performance_pipeline import (  # noqa: E501
            PipelineResult as _PR,  # noqa: N814
        )
        from llm_synthesis.services.pipelines.synthesis_performance_pipeline import (  # noqa: E501
            SynthesisWithPerformanceEntry as _SWP,  # noqa: N814
        )

        results = [
            _SWP(
                material=e.material,
                synthesis=e.synthesis,
                evaluation=e.evaluation,
                performance=performance_data.get(e.material),
                linking_evaluation=linking_evaluation,
            )
            for e in all_syntheses
        ]
        materials_with_perf = [m for m in materials if m in performance_data]
        materials_without_perf = [
            m for m in materials if m not in performance_data
        ]

        kept_relevant_plots: list = []
        if plots:
            kept_relevant_plots, _ = self._pipeline.plot_filter.filter_plots(
                plots, log_skipped=False
            )

        pipeline_result = _PR(
            paper_id=paper_id,
            paper_name=paper_id,
            materials=materials,
            results=results,
            plot_mappings=plot_mappings,
            num_plots=len(plots),
            linking_stats=linking_stats,
            materials_with_performance=materials_with_perf,
            materials_without_performance=materials_without_perf,
            relevant_plots=kept_relevant_plots,
            plot_figures=plot_figures,
        )

        cfg = self._domain_config_with_extractor
        processing_time = round(time.time() - paper_start, 1)

        summary = cfg.output_writer.write_paper(
            paper_id=paper_id,
            output_dir=output_dir,
            pipeline_result=pipeline_result,
            text_metrics={},
            vlm_metrics={},
            processing_time=processing_time,
        )
        logger.info(
            "  Phase 2 done: %d materials, %d plots, %.1fs",
            len(materials),
            len(plots),
            processing_time,
        )
        return summary

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _discover_papers(pdf_dir: Path) -> list[Path]:
        pdf_files = [
            p for p in sorted(pdf_dir.glob("*.pdf")) if not is_si_file(p)
        ]
        md_files = [
            p for p in sorted(pdf_dir.glob("*.md")) if not is_si_file(p)
        ]

        pdf_stems = {p.stem for p in pdf_files}
        all_papers = list(pdf_files)
        for md in md_files:
            if md.stem not in pdf_stems:
                all_papers.append(md)

        return sorted(all_papers, key=lambda p: p.name)

    @staticmethod
    def _filter_existing(papers: list[Path], output_dir: Path) -> list[Path]:
        remaining = []
        for p in papers:
            paper_dir = output_dir / p.stem
            done = (paper_dir / "linking_summary_llm.json").exists() or (
                paper_dir / "summary.json"
            ).exists()
            if done:
                logger.info("Skipping %s (already processed)", p.stem)
            else:
                remaining.append(p)
        return remaining

    @staticmethod
    def _print_batch_summary(
        all_summaries: list[dict],
        total_elapsed: float,
        output_dir: Path,
    ) -> None:
        n_ok = sum(1 for s in all_summaries if "error" not in s)
        n_fail = len(all_summaries) - n_ok
        print("\n" + "=" * 70)
        print("BATCH PROCESSING COMPLETE")
        print("=" * 70)
        print(f"  Papers processed: {n_ok}/{len(all_summaries)}")
        print(f"  Failed:           {n_fail}")
        print(
            f"  Total time:       {total_elapsed}s "
            f"({total_elapsed / 60:.1f} min)"
        )
        print()
        for s in all_summaries:
            if "error" in s:
                print(f"  [FAIL] {s['paper_id']}: {s['error']}")
            else:
                mats = s.get("total_materials", "?")
                perf = s.get("materials_with_performance", "?")
                t = s.get("processing_time_seconds", "?")
                print(
                    f"  [OK]   {s['paper_id']}: "
                    f"{mats} materials, {perf} with perf data, {t}s"
                )
        print(f"\n  Results: {output_dir}")
        print(f"  Batch summary: {output_dir / 'batch_summary.json'}")
