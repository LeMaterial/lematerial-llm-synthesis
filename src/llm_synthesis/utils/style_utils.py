"""Plotting utilities; colors, fonts, plot styles.

Usage:

from llm_synthesis.utils.style_utils import get_cmap, get_palette, set_style

set_style("manuscript") # Sets font, font size, figure size, color palette etc.
palette = get_palette() # Get the palette for the current style
cmap = get_cmap() # Get the cmap for the current style

"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# from cmcrameri import cm

# cmap = cm.batlow
# palette = cmap.colors
# # convert to list
# palette = list(palette)
# palette.append("#D3D3D3")
# palette.append("#A9A9A9")
# palette.append("#808080")

# initialize empty palette
palette = []
palette.append("#E0A2D3")
palette.append("#DFD6FB")
palette.append("#448FF2")
palette.append("#A7C8F2")
palette.append("#F9CB9C")
palette.append("#FCE5CD")
palette.append("#FCEAF7")
palette.append("#F2A8DD")
palette.append("#D3D3D3")

palette.append("#A9A9A9")

palette.append("#808080")
palette.append("#000000")
palette.append("#7B5AEF")
palette.append("#B27EDD")
palette.append("#2E8B57")
palette.append("#C97B3D")

sns.set_palette(palette)

# create cmap out of palette
cmap = mpl.colors.ListedColormap(palette)


def get_palette() -> list[str]:
    """Get the palette for the current style.

    Returns:
      : list[str]: Palette colors
    """
    return palette


def get_cmap() -> list[str]:
    """Get the cmap for the current style.

    Returns:
      : list[str]: Cmap colors
    """
    return cmap


# Font types, font sizes, plot sizes etc.
def set_style(style: str = "manuscript") -> None:
    """Set the matplotlib style parameters based on the chosen style.

    Args:
      style: str: Style to use (manuscript, presentation, poster)

    Returns:
      : None
    """
    # Common settings for all styles
    common_settings = {
        "pdf.fonttype": 42,
        "font.family": "DejaVu Sans",
        "mathtext.fontset": "dejavusans",
        "text.latex.preamble": (
            r"\usepackage{amsmath} \usepackage{amssymb} \usepackage{sfmath}"
        ),
        # Figure size
        "figure.figsize": (3.3, 2.5),
        "figure.dpi": 300,
        "figure.facecolor": "white",
        # Tick parameters
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.minor.size": 1.5,
        "ytick.minor.size": 1.5,
        "xtick.minor.width": 0.5,
        "ytick.minor.width": 0.5,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "xtick.top": False,
        "ytick.right": False,
        "xtick.minor.top": False,
        "xtick.minor.bottom": False,
        "ytick.minor.left": False,
        # Line and marker settings
        "axes.linewidth": 0.5,
        "grid.linewidth": 0.5,
        "lines.linewidth": 1.0,
        "lines.markersize": 3,
        "hatch.linewidth": 0.5,
        # Margins
        "axes.xmargin": 0,
        "axes.ymargin": 0,
        # Remove spines
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.axisbelow": True,
        # Legend settings
        "legend.frameon": False,
        "legend.fancybox": False,
        "legend.facecolor": "none",
        # Color cycle
        "axes.prop_cycle": plt.cycler(
            "color",
            [
                "#0C5DA5",
                "#00B945",
                "#FF9500",
                "#FF2C00",
                "#845B97",
                "#474747",
                "#9e9e9e",
                "#9A607F",
            ],
        ),
        "text.color": "black",
        "axes.labelcolor": "black",
        "axes.edgecolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",
    }

    # Style-specific settings
    if style == "manuscript":
        style_settings = {
            # Font sizes
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            # Tick sizes
            "xtick.major.size": 3,
            "ytick.major.size": 3,
        }
    elif style == "presentation":
        style_settings = {
            # Font sizes (larger for presentations)
            "font.size": 12,
            "axes.labelsize": 13,
            "axes.titlesize": 13,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            # Tick sizes
            "xtick.major.size": 3,
            "ytick.major.size": 3,
        }
    elif style == "poster":
        style_settings = {
            # Font sizes
            "font.size": 12,
            "axes.labelsize": 13,
            "axes.titlesize": 13,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            # Tick sizes
            "xtick.major.size": 3,
            "ytick.major.size": 3,
        }
    else:
        raise KeyError(
            f"Style '{style}' not recognized.",
            "Available styles: 'manuscript', 'presentation', 'poster'",
        )

    # Apply settings
    settings = {**common_settings, **style_settings}
    for key, value in settings.items():
        mpl.rcParams[key] = value


SOURCE_ORDER = ["arxiv", "chemrxiv", "omg24"]

BORDER_STYLE = {
    "axes.spines.top": True,
    "axes.spines.right": True,
    "axes.linewidth": 1.4,
    "xtick.major.width": 1.2,
    "ytick.major.width": 1.2,
    "xtick.major.size": 4,
    "ytick.major.size": 4,
}


def save_fig(fig, fig_dir: Path, name: str, extra_artists=None) -> None:
    """Save a figure as both SVG and PDF under fig_dir."""
    kwargs = {"bbox_inches": "tight"}
    if extra_artists:
        kwargs["bbox_extra_artists"] = extra_artists
    fig.savefig(Path(fig_dir) / f"{name}.svg", **kwargs)
    fig.savefig(Path(fig_dir) / f"{name}.pdf", **kwargs)


def plot_category_bar(
    df,
    column,
    top_n=None,
    stack_by=None,
    drop=None,
    name=None,
    fig_dir=None,
    log_scale=False,
):
    """Horizontal bar chart of `column` value counts, house style.

    If stack_by is given (e.g. "source" or "material_category"), splits each
    bar by that column's values; otherwise each bar gets its own palette
    color, with count labels. drop excludes given category values (e.g.
    "other") before ranking. top_n=None plots every remaining category
    (appendix use), otherwise only the top_n most frequent (pub use).
    Largest category on top via invert_yaxis. log_scale=True switches the
    count axis to log scale (bar_label counts still shown as-is).
    """
    if drop:
        df = df[~df[column].isin(drop)]

    order = df[column].value_counts().index.tolist()
    if top_n is not None:
        order = order[:top_n]

    row_height = 0.4 if top_n is not None else 0.22
    with plt.rc_context(BORDER_STYLE):
        fig, ax = plt.subplots(figsize=(6, row_height * len(order) + 1))

        if stack_by:
            stack_order = (
                SOURCE_ORDER
                if stack_by == "source"
                else df[stack_by].value_counts().index.tolist()
            )
            pivot = (
                df[df[column].isin(order)]
                .groupby([column, stack_by])
                .size()
                .unstack(fill_value=0)
                .reindex(order)
            )
            pivot = pivot[[c for c in stack_order if c in pivot.columns]]
            pivot.plot(
                kind="barh",
                stacked=True,
                ax=ax,
                color=palette[: len(pivot.columns)],
                width=0.6,
            )
            legend = ax.legend(
                title=stack_by.replace("_", " ").title(),
                bbox_to_anchor=(0.5, -0.2),
                loc="upper center",
                ncol=3,
                frameon=False,
            )
        else:
            counts = df[column].value_counts().reindex(order)
            bar_colors = [palette[i % len(palette)] for i in range(len(order))]
            bars = ax.barh(
                counts.index, counts.values, color=bar_colors, height=0.6
            )
            ax.bar_label(bars, fmt="%d", padding=3, fontsize=8)

        if log_scale:
            ax.set_xscale("log")

        ax.set_xlabel("# Materials")
        ax.set_ylabel(column.replace("_", " ").title())
        ax.invert_yaxis()

        if stack_by:
            # tight_layout doesn't know about the below-axes legend; let
            # bbox_inches="tight" at save time size the canvas instead.
            fig.canvas.draw()
        else:
            fig.tight_layout()

        if name and fig_dir:
            extra = (legend,) if stack_by else None
            save_fig(fig, fig_dir, name, extra_artists=extra)
    return fig, ax
