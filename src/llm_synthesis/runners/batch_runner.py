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
import traceback
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
from llm_synthesis.transformers.synthesis_extraction.dspy_synthesis_extraction import (  # noqa: E501
    DspySynthesisExtractor,
    make_dspy_synthesis_extractor_signature,
)
from llm_synthesis.utils.dspy_utils import get_llm_from_name
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
    ) -> None:
        self.domain_config = domain_config
        self.gemini_model = gemini_model
        self.claude_model = claude_model
        self.linker_model = linker_model
        self.material_model = material_model or gemini_model
        self.synthesis_max_tokens = synthesis_max_tokens
        self.linker_max_tokens = linker_max_tokens

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
    ) -> None:
        """Process every PDF in pdf_dir and write results to output_dir.

        Args:
            pdf_dir: Directory containing PDF (and optionally MD/TXT) files.
            output_dir: Root directory for output files.
            max_papers: If set, stop after this many papers (useful for
                testing).
            skip_existing: If True, skip papers whose output already exists.
            skip_figures: If True, skip figure/plot extraction entirely.
        """
        import warnings

        warnings.filterwarnings(
            "ignore", category=UserWarning, module="pydantic"
        )
        logging.getLogger("LiteLLM").setLevel(logging.ERROR)
        logging.getLogger("litellm").setLevel(logging.ERROR)
        logging.getLogger("pydantic").setLevel(logging.ERROR)

        pdf_dir = Path(pdf_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

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

        # Plot extractor (Claude VLM)
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
