"""Command-line interface for LeMat-Synth.

Provides two commands:

    lemat-synth extract <input>    [key=value ...]   # one paper (PDF or text)
    lemat-synth batch   <input-dir> [key=value ...]  # folder of papers

Configuration is driven by ``config/cli.yaml`` at the repository root.
Every setting can be overridden on the command line using Hydra key=value
syntax.  Examples::

    lemat-synth extract paper.txt synthesis_model=anthropic/claude-sonnet-4-6
    lemat-synth batch papers/ domain=catalysis skip_existing=false
    lemat-synth extract paper.txt \\
        "prompts.synthesis_instructions=Extract only the main route."

For a full list of configurable keys see ``config/cli.yaml`` or the CLI
reference in the documentation.
"""

import asyncio
import logging
import os
import warnings
from pathlib import Path
from typing import Annotated, Any

import typer
from dotenv import load_dotenv
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from llm_synthesis.config.plot_filter_config import PlotFilterConfig
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
from llm_synthesis.transformers.pdf_extraction import (
    DoclingPDFExtractor,
    MistralPDFExtractor,
)
from llm_synthesis.transformers.performance_linking import (
    series_material_linker,
)
from llm_synthesis.transformers.plot_extraction.claude_extraction import (
    plot_data_extraction as claude_plot_data,
)
from llm_synthesis.transformers.synthesis_extraction.dspy_synthesis_extraction import (  # noqa: E501
    DspySynthesisExtractor,
    make_dspy_synthesis_extractor_signature,
)
from llm_synthesis.utils.concurrency import get_max_concurrent_llm_calls
from llm_synthesis.utils.llms import SystemPrefixedLM

# Valid environment variable names that hold LLM API keys. Users select one
# of these by name (e.g. synthesis_api_key_env=OPENROUTER_QWEN_API_KEY) — the
# actual key value is never passed on the command line.
_ALLOWED_API_KEY_ENVS: frozenset[str] = frozenset({
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
    "OPENAI_API_KEY",
    "MISTRAL_API_KEY",
    "OPENROUTER_QWEN_API_KEY",
    "OPENROUTER_KIMI_API_KEY",
    "OPENROUTER_DEEPSEEK_API_KEY",
})

# Silence noisy third-party loggers
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
logging.getLogger("pydantic").setLevel(logging.ERROR)
logging.getLogger("LiteLLM").setLevel(logging.ERROR)
logging.getLogger("litellm").setLevel(logging.ERROR)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

app = typer.Typer(
    name="lemat-synth",
    help=(
        "LeMat-Synth: extract structured synthesis procedures from "
        "materials science papers.\n\n"
        "Configuration is read from config/cli.yaml at the repository "
        "root and can be overridden with Hydra key=value arguments.\n\n"
        "Run 'lemat-synth extract --help' or 'lemat-synth batch --help' "
        "for per-command usage."
    ),
    add_completion=False,
)


# ── Environment ───────────────────────────────────────────────────────────────


def _load_env() -> None:
    """Load .env from the repository root (three levels above this file)."""
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=True)
    else:
        load_dotenv(override=True)


# ── API key resolver ──────────────────────────────────────────────────────────


def _resolve_api_key(env_var_name: str | None) -> str | None:
    """Return the value of a known API-key env var, or None if not specified.

    Raises ValueError when env_var_name is not in the allowed set, so typos
    fail loudly rather than silently falling back to auto-detection.
    """
    if env_var_name is None:
        return None
    if env_var_name not in _ALLOWED_API_KEY_ENVS:
        raise ValueError(
            f"Unknown api_key_env '{env_var_name}'. "
            f"Allowed values: {', '.join(sorted(_ALLOWED_API_KEY_ENVS))}"
        )
    return os.environ.get(env_var_name)


# ── Hydra config ──────────────────────────────────────────────────────────────


def _load_cli_cfg(overrides: list[str]) -> Any:
    """Load config/cli.yaml and apply Hydra key=value overrides."""
    GlobalHydra.instance().clear()
    config_dir = str(Path(__file__).resolve().parents[2] / "config")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        return compose(config_name="cli", overrides=overrides)


# ── LM builder ────────────────────────────────────────────────────────────────


def _build_lm(
    model: str,
    system_prompt: str = "",
    api_base: str | None = None,
    api_key_env: str | None = None,
    **kwargs,
) -> SystemPrefixedLM:
    """Build a SystemPrefixedLM for any LiteLLM-compatible model string."""
    if api_base is not None:
        kwargs["api_base"] = api_base
    api_key = _resolve_api_key(api_key_env)
    if api_key is not None:
        kwargs["api_key"] = api_key
    return SystemPrefixedLM(system_prompt, model, **kwargs)


# ── Pipeline builder ──────────────────────────────────────────────────────────


def _build_pipeline_from_cfg(cfg: Any) -> SynthesisPerformancePipeline:
    """Build a SynthesisPerformancePipeline from a loaded Hydra DictConfig."""
    api_base = OmegaConf.select(cfg, "api_base", default=None)
    p = cfg.prompts

    # ── Material extractor ────────────────────────────────────────────────────
    material_sig = make_dspy_text_extractor_signature(
        signature_name="MaterialsExtraction",
        instructions=p.material_instructions,
        input_description=p.material_input_description,
        output_name="materials",
        output_description=p.material_output_description,
    )
    material_lm = _build_lm(
        cfg.material_model,
        api_base=api_base,
        api_key_env=OmegaConf.select(
            cfg, "material_api_key_env", default=None
        ),
        temperature=0.0,
    )
    material_extractor = DspyTextExtractor(
        signature=material_sig, lm=material_lm
    )

    # ── Synthesis extractor ───────────────────────────────────────────────────
    synthesis_sig = make_dspy_synthesis_extractor_signature(
        signature_name="StructuredSynthesisExtraction",
        instructions=p.synthesis_instructions,
        paper_text_description=p.paper_text_description,
        material_name_description=p.material_name_description,
        output_name="structured_synthesis",
        output_description=p.synthesis_output_description,
    )
    synthesis_lm = _build_lm(
        cfg.synthesis_model,
        system_prompt=p.synthesis_system,
        api_base=api_base,
        api_key_env=OmegaConf.select(
            cfg, "synthesis_api_key_env", default=None
        ),
        temperature=0.0,
        max_tokens=8000,
        num_retries=3,
    )
    synthesis_extractor = DspySynthesisExtractor(
        signature=synthesis_sig, lm=synthesis_lm
    )

    # ── Quality judge ─────────────────────────────────────────────────────────
    judge_lm = _build_lm(
        cfg.judge_model,
        api_base=api_base,
        api_key_env=OmegaConf.select(
            cfg, "judge_api_key_env", default=None
        ),
        temperature=0.1,
        max_tokens=8000,
    )
    judge = DspyGeneralSynthesisJudge(
        signature=make_general_synthesis_judge_signature(), lm=judge_lm
    )

    # ── Plot filter ───────────────────────────────────────────────────────────
    domain_map = {
        "catalysis": PlotFilterConfig.for_catalysis,
        "superconductors": PlotFilterConfig.for_superconductivity,
        "electrochemistry": PlotFilterConfig.for_electrochemistry,
        "generic": PlotFilterConfig.no_filter,
    }
    plot_filter_config = domain_map.get(
        cfg.domain, PlotFilterConfig.no_filter
    )()

    # ── Performance components (optional) ─────────────────────────────────────
    plot_extractor = None
    series_linker = None
    linking_judge = None

    if cfg.with_performance:
        plot_extractor = claude_plot_data.ClaudeLinePlotDataExtractor(
            model_name=cfg.plot_model
        )
        linker_lm = _build_lm(
            cfg.linker_model,
            api_base=api_base,
            api_key_env=OmegaConf.select(
                cfg, "linker_api_key_env", default=None
            ),
            temperature=0.0,
            max_tokens=8000,
        )
        series_linker = series_material_linker.SeriesMaterialLinker(
            lm=linker_lm
        )
        linking_judge_lm = _build_lm(
            cfg.judge_model,
            api_base=api_base,
            api_key_env=OmegaConf.select(
                cfg, "judge_api_key_env", default=None
            ),
            temperature=0.1,
            max_tokens=8000,
        )
        linking_judge = DspyLinkingJudge(
            signature=make_linking_judge_signature(), lm=linking_judge_lm
        )

    return SynthesisPerformancePipeline(
        material_extractor=material_extractor,
        synthesis_extractor=synthesis_extractor,
        judge=judge,
        linking_judge=linking_judge,
        plot_extractor=plot_extractor,
        series_linker=series_linker,
        plot_filter_config=plot_filter_config,
        figure_segmenter=cfg.figure_segmenter,
        florence_repo_id=cfg.florence_repo_id,
    )


# ── File helpers ──────────────────────────────────────────────────────────────


def _load_paper_from_file(path: Path) -> Paper:
    """Load a single paper from a .txt or .md file as a Paper object."""
    text = path.read_text(encoding="utf-8", errors="replace")
    si_path = path.parent / (path.stem + "_SI" + path.suffix)
    si_text = (
        si_path.read_text(encoding="utf-8", errors="replace")
        if si_path.exists()
        else ""
    )
    return Paper(
        name=path.stem, id=path.stem, publication_text=text, si_text=si_text
    )


def _save_result(result: Any, output_dir: Path) -> None:
    """Save a PipelineResult to <output_dir>/<paper_id>/."""
    SynthesisPerformancePipeline.save_results(result, str(output_dir))


def _pdf_to_markdown(
    pdf_path: Path,
    output_dir: Path,
    extractor: str = "docling",
) -> Path:
    """Extract markdown text from a PDF and return the path to the .md file."""
    if extractor == "mistral":
        pdf_extractor = MistralPDFExtractor(structured=False)
    else:
        pdf_extractor = DoclingPDFExtractor()
    md_text = pdf_extractor.forward(pdf_path.read_bytes())
    md_path = output_dir / "_extracted_text" / (pdf_path.stem + ".md")
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(md_text, encoding="utf-8")
    typer.echo(f"  -> Extracted {len(md_text):,} characters.")
    return md_path


def _llm_semaphore() -> asyncio.Semaphore:
    """Return a semaphore sized from the project-wide env-driven default."""
    return asyncio.Semaphore(get_max_concurrent_llm_calls())


# ── Commands ──────────────────────────────────────────────────────────────────


@app.command(
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True}
)
def extract(
    ctx: typer.Context,
    input_file: Annotated[
        Path,
        typer.Argument(
            help=(
                "Path to a single paper: a .txt or .md text file, "
                "or a .pdf file."
            )
        ),
    ],
) -> None:
    """Extract synthesis procedures from a single paper.

    Settings are read from config/cli.yaml and can be overridden with
    Hydra key=value arguments passed after the file path.

    \b
    Examples:
      lemat-synth extract paper.txt
      lemat-synth extract paper.pdf output_dir=my_results/
      lemat-synth extract paper.txt synthesis_model=anthropic/claude-sonnet-4-6
      lemat-synth extract paper.txt domain=catalysis with_performance=true
      lemat-synth extract paper.pdf pdf_extractor=mistral
      lemat-synth extract paper.txt \\
          synthesis_model=openrouter/google/gemini-3-flash-preview \\
          api_base=https://openrouter.ai/api/v1
    """
    _load_env()
    cfg = _load_cli_cfg(ctx.args)

    if not input_file.exists():
        typer.echo(f"ERROR: File not found: {input_file}", err=True)
        raise typer.Exit(1)

    output = Path(cfg.output_dir)

    # PDF → text if needed
    paper_text_file = input_file
    if input_file.suffix.lower() == ".pdf":
        typer.echo(f"Extracting text from PDF: {input_file.name} ...")
        paper_text_file = _pdf_to_markdown(
            input_file, output, extractor=cfg.pdf_extractor
        )

    typer.echo(f"Loading paper: {paper_text_file.name}")
    paper = _load_paper_from_file(paper_text_file)

    typer.echo(
        f"Building pipeline "
        f"(synthesis_model={cfg.synthesis_model}, domain={cfg.domain}) ..."
    )
    pipeline = _build_pipeline_from_cfg(cfg)

    output.mkdir(parents=True, exist_ok=True)
    semaphore = _llm_semaphore()

    typer.echo("Running extraction ...")
    result = asyncio.run(
        pipeline.process_paper_async(
            paper, semaphore, skip_figures=not cfg.with_performance
        )
    )

    if result is None:
        typer.echo(
            "No results produced. "
            "The paper may contain no synthesis procedures."
        )
        raise typer.Exit(0)

    _save_result(result, output)

    typer.echo(
        f"\nDone. Found {len(result.materials)} material(s): "
        + ", ".join(result.materials)
    )
    typer.echo(f"Results written to: {output / result.paper_id}/")

    if cfg.with_performance and result.linking_stats:
        s = result.linking_stats
        typer.echo(
            f"Performance: {s.plots_linked}/{s.total_plots_extracted} "
            "plots linked."
        )


@app.command(
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True}
)
def batch(
    ctx: typer.Context,
    input_dir: Annotated[
        Path,
        typer.Argument(
            help="Folder containing .txt, .md, or .pdf paper files."
        ),
    ],
) -> None:
    """Extract synthesis procedures from a folder of papers.

    Settings are read from config/cli.yaml and can be overridden with
    Hydra key=value arguments passed after the directory path.

    \b
    Examples:
      lemat-synth batch papers/
      lemat-synth batch papers/ output_dir=results/ domain=catalysis
      lemat-synth batch papers/ synthesis_model=gemini/gemini-2.5-pro \\
          max_papers=10
      lemat-synth batch papers/ skip_existing=false max_papers_parallel=2
      lemat-synth batch papers/ pdf_extractor=mistral
    """
    _load_env()
    cfg = _load_cli_cfg(ctx.args)

    if not input_dir.is_dir():
        typer.echo(f"ERROR: Not a directory: {input_dir}", err=True)
        raise typer.Exit(1)

    output_dir = Path(cfg.output_dir)
    max_papers: int | None = cfg.max_papers
    skip_existing: bool = cfg.skip_existing
    max_papers_parallel: int = cfg.max_papers_parallel

    paper_files = sorted(
        list(input_dir.glob("*.txt"))
        + list(input_dir.glob("*.md"))
        + list(input_dir.glob("*.pdf"))
    )
    if not paper_files:
        typer.echo(f"No .txt / .md / .pdf files found in {input_dir}.")
        raise typer.Exit(0)

    if max_papers:
        paper_files = paper_files[:max_papers]

    typer.echo(f"Found {len(paper_files)} papers in {input_dir}.")
    typer.echo(
        f"Building pipeline "
        f"(synthesis_model={cfg.synthesis_model}, domain={cfg.domain}) ..."
    )
    pipeline = _build_pipeline_from_cfg(cfg)

    output_dir.mkdir(parents=True, exist_ok=True)
    semaphore = _llm_semaphore()
    paper_semaphore = asyncio.Semaphore(max_papers_parallel)

    async def _process_one(path: Path) -> None:
        paper_id = path.stem
        paper_out = output_dir / paper_id
        if (
            skip_existing
            and paper_out.is_dir()
            and any(paper_out.glob("*.json"))
        ):
            typer.echo(f"  Skipping {paper_id} (already processed)")
            return

        if path.suffix.lower() == ".pdf":
            paper_path = _pdf_to_markdown(
                path, output_dir, extractor=cfg.pdf_extractor
            )
        else:
            paper_path = path

        paper = _load_paper_from_file(paper_path)

        async with paper_semaphore:
            try:
                result = await pipeline.process_paper_async(
                    paper, semaphore, skip_figures=not cfg.with_performance
                )
                if result:
                    _save_result(result, output_dir)
                    typer.echo(
                        f"  {paper_id}: {len(result.materials)} "
                        "material(s) — "
                        + ", ".join(result.materials)
                    )
                else:
                    typer.echo(f"  {paper_id}: no synthesis found")
            except Exception as exc:
                logger.exception("Failed processing %s", paper_id)
                typer.echo(
                    f"  ERROR processing {paper_id}: {exc}", err=True
                )

    async def _run_all() -> None:
        await asyncio.gather(*[_process_one(p) for p in paper_files])

    asyncio.run(_run_all())
    typer.echo(f"\nBatch complete. Results written to: {output_dir}/")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
