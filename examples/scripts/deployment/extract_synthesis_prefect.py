"""Prefect-orchestrated synthesis extraction entry point.

Replaces ThreadPoolExecutor-based parallelism with Prefect flow/task
orchestration, while preserving Hydra configuration management.

Usage:
    uv run python examples/scripts/deployment/extract_synthesis_prefect.py
    uv run python examples/scripts/deployment/extract_synthesis_prefect.py \
        data_loader.number_of_samples=10
"""

import logging
import os
import warnings

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
from prefect.task_runners import ThreadPoolTaskRunner

from llm_synthesis.orchestration.flows import synthesis_extraction_flow

# Suppress noisy third-party warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
logging.getLogger("pydantic").setLevel(logging.ERROR)
logging.getLogger("LiteLLM").setLevel(logging.ERROR)
logging.getLogger("litellm").setLevel(logging.ERROR)


def _resolve_path(base: str, path: str) -> str:
    """Resolve a relative path; leaves absolute/cloud paths unchanged."""
    if path.startswith(("s3://", "gs://", "/")):
        return path
    return os.path.join(base, path)


@hydra.main(
    config_path="../../config", config_name="config.yaml", version_base=None
)
def main(cfg: DictConfig) -> None:
    original_cwd = get_original_cwd()

    # Resolve all relative paths to absolute BEFORE converting to plain dict.
    # Hydra changes CWD to the output directory; Prefect needs absolute paths.
    if hasattr(cfg.data_loader, "architecture") and hasattr(
        cfg.data_loader.architecture, "data_dir"
    ):
        cfg.data_loader.architecture.data_dir = _resolve_path(
            original_cwd, cfg.data_loader.architecture.data_dir
        )

    if hasattr(cfg.material_extraction, "architecture"):
        arch = cfg.material_extraction.architecture
        if hasattr(arch, "lm") and hasattr(arch.lm, "system_prompt"):
            sp = arch.lm.system_prompt
            if hasattr(sp, "prompt_path"):
                sp.prompt_path = _resolve_path(original_cwd, sp.prompt_path)

    if hasattr(cfg.synthesis_extraction, "architecture"):
        arch = cfg.synthesis_extraction.architecture
        if hasattr(arch, "lm") and hasattr(arch.lm, "system_prompt"):
            sp = arch.lm.system_prompt
            if hasattr(sp, "prompt_path"):
                sp.prompt_path = _resolve_path(original_cwd, sp.prompt_path)

    if hasattr(cfg.result_save, "architecture") and hasattr(
        cfg.result_save.architecture, "result_dir"
    ):
        cfg.result_save.architecture.result_dir = _resolve_path(
            original_cwd, cfg.result_save.architecture.result_dir
        )

    # Convert to plain serializable dict for Prefect flow.
    # Prefect cannot serialize OmegaConf DictConfig or non-picklable objects.
    config_dict = OmegaConf.to_container(cfg, resolve=True)

    max_workers: int = (
        config_dict.get("orchestration", {}).get("max_workers", 4)  # type: ignore[union-attr]
    )
    synthesis_extraction_flow.with_options(
        task_runner=ThreadPoolTaskRunner(max_workers=max_workers)
    )(config_dict)


if __name__ == "__main__":
    main()
