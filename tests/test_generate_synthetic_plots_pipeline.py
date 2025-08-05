import importlib.util
import random
import sys
import types
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import PathPatch

BASE_DIR = Path(__file__).resolve().parents[1] / "src" / "llm_synthesis"
PLOTS_DIR = Path(__file__).resolve().parents[1] / "tests" / "test_generate_synthetic_plots_pipeline_plots"
PLOTS_DIR.mkdir(exist_ok=True)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    sys.modules[name] = module
    return module


# Create namespace packages to satisfy imports
sys.modules.setdefault("llm_synthesis", types.ModuleType("llm_synthesis"))
sys.modules.setdefault(
    "llm_synthesis.services", types.ModuleType("llm_synthesis.services")
)
sys.modules.setdefault(
    "llm_synthesis.services.pipelines",
    types.ModuleType("llm_synthesis.services.pipelines"),
)
sys.modules.setdefault(
    "llm_synthesis.utils", types.ModuleType("llm_synthesis.utils")
)

# Load dependent modules
_load_module(
    "llm_synthesis.services.pipelines.base_pipeline",
    BASE_DIR / "services" / "pipelines" / "base_pipeline.py",
)
_load_module(
    "llm_synthesis.utils.synthetic_figure_utils",
    BASE_DIR / "utils" / "synthetic_figure_utils.py",
)

# Finally, load the pipeline module
GenerateSyntheticPlotsPipeline = _load_module(
    "llm_synthesis.services.pipelines.generate_synthetic_plots_pipeline",
    BASE_DIR
    / "services"
    / "pipelines"
    / "generate_synthetic_plots_pipeline.py",
).GenerateSyntheticPlotsPipeline


def test_scatter_plot_without_lines(tmp_path):
    random.seed(0)
    np.random.seed(0)
    pipeline = GenerateSyntheticPlotsPipeline(
        num_plots=1, images_path=str(tmp_path), groundtruths_path=str(tmp_path)
    )
    fig, ax = plt.subplots()
    coords, _, _ = pipeline._plot_random_scatter_on_ax(ax)
    try:
        fig.savefig(PLOTS_DIR / "scatter_no_lines.png", bbox_inches="tight")
        assert (PLOTS_DIR / "scatter_no_lines.png").exists()
        assert len(ax.lines) == 0
        assert len(ax.collections) > 0
        assert coords
    finally:
        plt.close(fig)


def test_scatter_plot_with_lines(tmp_path):
    random.seed(1)
    np.random.seed(1)
    pipeline = GenerateSyntheticPlotsPipeline(
        num_plots=1, images_path=str(tmp_path), groundtruths_path=str(tmp_path)
    )
    fig, ax = plt.subplots()
    coords, _, _ = pipeline._plot_random_scatter_on_ax(ax)
    try:
        fig.savefig(PLOTS_DIR / "scatter_with_lines.png", bbox_inches="tight")
        assert (PLOTS_DIR / "scatter_with_lines.png").exists()
        assert len(ax.lines) > 0
        assert len(ax.collections) > 0
        assert coords
    finally:
        plt.close(fig)


def test_boxplot_generation(tmp_path):
    random.seed(2)
    np.random.seed(2)
    pipeline = GenerateSyntheticPlotsPipeline(
        num_plots=1, images_path=str(tmp_path), groundtruths_path=str(tmp_path)
    )
    fig, ax = plt.subplots()
    coords, _, _ = pipeline._plot_random_boxplot_on_ax(ax)
    try:
        fig.savefig(PLOTS_DIR / "box_plot.png", bbox_inches="tight")
        assert (PLOTS_DIR / "box_plot.png").exists()
        boxes = [p for p in ax.patches if isinstance(p, PathPatch)]
        assert len(boxes) >= 1
        assert coords
    finally:
        plt.close(fig)


def test_pipeline_generates_multiple_plots():
    random.seed(3)
    np.random.seed(3)
    pipeline = GenerateSyntheticPlotsPipeline(
        num_plots=2,
        images_path=str(PLOTS_DIR),
        groundtruths_path=str(PLOTS_DIR),
    )
    for i in range(2):
        img = PLOTS_DIR / f"figure_{i}.png"
        gt = PLOTS_DIR / f"figure_{i}.json"
        if img.exists():
            img.unlink()
        if gt.exists():
            gt.unlink()
    pipeline.run()
    for i in range(2):
        assert (PLOTS_DIR / f"figure_{i}.png").exists()
        assert (PLOTS_DIR / f"figure_{i}.json").exists()



def generate_synthetic_plots_example() -> None:
    """Generate sample synthetic plots for each supported type."""
    import random
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np

    from llm_synthesis.services.pipelines.generate_synthetic_plots_pipeline import (  # noqa: E501
        GenerateSyntheticPlotsPipeline,
    )

    out_dir = Path(PLOTS_DIR / "synthetic_plot_examples")
    out_dir.mkdir(exist_ok=True)
    pipeline = GenerateSyntheticPlotsPipeline(
        num_plots=1,
        images_path=str(out_dir),
        groundtruths_path=str(out_dir),
    )

    random.seed(0)
    np.random.seed(0)
    fig, ax = plt.subplots()
    pipeline._plot_random_scatter_on_ax(ax)
    fig.savefig(out_dir / "scatter_no_lines.png", bbox_inches="tight")
    plt.close(fig)

    random.seed(1)
    np.random.seed(1)
    fig, ax = plt.subplots()
    pipeline._plot_random_scatter_on_ax(ax)
    fig.savefig(out_dir / "scatter_with_lines.png", bbox_inches="tight")
    plt.close(fig)

    random.seed(2)
    np.random.seed(2)
    fig, ax = plt.subplots()
    pipeline._plot_random_boxplot_on_ax(ax)
    fig.savefig(out_dir / "box_plot.png", bbox_inches="tight")
    plt.close(fig)

    random.seed(3)
    np.random.seed(3)
    pipeline._plot_multiple_subplots(
        out_dir / "multi_subplot.png", out_dir / "multi_subplot.json"
    )

import importlib.util
import random
import sys
import types
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import PathPatch

BASE_DIR = Path(__file__).resolve().parents[1] / "src" / "llm_synthesis"
PLOTS_DIR = Path(__file__).resolve().parents[1] / "plots"
PLOTS_DIR.mkdir(exist_ok=True)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    sys.modules[name] = module
    return module


# Create namespace packages to satisfy imports
sys.modules.setdefault("llm_synthesis", types.ModuleType("llm_synthesis"))
sys.modules.setdefault(
    "llm_synthesis.services", types.ModuleType("llm_synthesis.services")
)
sys.modules.setdefault(
    "llm_synthesis.services.pipelines",
    types.ModuleType("llm_synthesis.services.pipelines"),
)
sys.modules.setdefault(
    "llm_synthesis.utils", types.ModuleType("llm_synthesis.utils")
)

# Load dependent modules
_load_module(
    "llm_synthesis.services.pipelines.base_pipeline",
    BASE_DIR / "services" / "pipelines" / "base_pipeline.py",
)
_load_module(
    "llm_synthesis.utils.synthetic_figure_utils",
    BASE_DIR / "utils" / "synthetic_figure_utils.py",
)

# Finally, load the pipeline module
GenerateSyntheticPlotsPipeline = _load_module(
    "llm_synthesis.services.pipelines.generate_synthetic_plots_pipeline",
    BASE_DIR
    / "services"
    / "pipelines"
    / "generate_synthetic_plots_pipeline.py",
).GenerateSyntheticPlotsPipeline


def test_scatter_plot_without_lines(tmp_path):
    random.seed(0)
    np.random.seed(0)
    pipeline = GenerateSyntheticPlotsPipeline(
        num_plots=1, images_path=str(tmp_path), groundtruths_path=str(tmp_path)
    )
    fig, ax = plt.subplots()
    coords, _, _ = pipeline._plot_random_scatter_on_ax(ax)
    try:
        fig.savefig(PLOTS_DIR / "scatter_no_lines.png", bbox_inches="tight")
        assert (PLOTS_DIR / "scatter_no_lines.png").exists()
        assert len(ax.lines) == 0
        assert len(ax.collections) > 0
        assert coords
    finally:
        plt.close(fig)


def test_scatter_plot_with_lines(tmp_path):
    random.seed(1)
    np.random.seed(1)
    pipeline = GenerateSyntheticPlotsPipeline(
        num_plots=1, images_path=str(tmp_path), groundtruths_path=str(tmp_path)
    )
    fig, ax = plt.subplots()
    coords, _, _ = pipeline._plot_random_scatter_on_ax(ax)
    try:
        fig.savefig(PLOTS_DIR / "scatter_with_lines.png", bbox_inches="tight")
        assert (PLOTS_DIR / "scatter_with_lines.png").exists()
        assert len(ax.lines) > 0
        assert len(ax.collections) > 0
        assert coords
    finally:
        plt.close(fig)


def test_boxplot_generation(tmp_path):
    random.seed(2)
    np.random.seed(2)
    pipeline = GenerateSyntheticPlotsPipeline(
        num_plots=1, images_path=str(tmp_path), groundtruths_path=str(tmp_path)
    )
    fig, ax = plt.subplots()
    coords, _, _ = pipeline._plot_random_boxplot_on_ax(ax)
    try:
        fig.savefig(PLOTS_DIR / "box_plot.png", bbox_inches="tight")
        assert (PLOTS_DIR / "box_plot.png").exists()
        boxes = [p for p in ax.patches if isinstance(p, PathPatch)]
        assert len(boxes) >= 1
        assert coords
    finally:
        plt.close(fig)


def test_pipeline_generates_multiple_plots():
    random.seed(3)
    np.random.seed(3)
    pipeline = GenerateSyntheticPlotsPipeline(
        num_plots=2,
        images_path=str(PLOTS_DIR),
        groundtruths_path=str(PLOTS_DIR),
    )
    for i in range(2):
        img = PLOTS_DIR / f"figure_{i}.png"
        gt = PLOTS_DIR / f"figure_{i}.json"
        if img.exists():
            img.unlink()
        if gt.exists():
            gt.unlink()
    pipeline.run()
    for i in range(2):
        assert (PLOTS_DIR / f"figure_{i}.png").exists()
        assert (PLOTS_DIR / f"figure_{i}.json").exists()

if __name__ == "__main__":
    from llm_synthesis.services.pipelines.generate_synthetic_plots_pipeline import GenerateSyntheticPlotsPipeline
    p = GenerateSyntheticPlotsPipeline(num_plots=8, images_path=PLOTS_DIR, groundtruths_path=PLOTS_DIR)
    p.run()
    # generate_synthetic_plots_example()
    print(f"  Saved plots and labels to `{PLOTS_DIR}`")