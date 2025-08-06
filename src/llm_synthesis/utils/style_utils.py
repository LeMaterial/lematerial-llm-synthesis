"""Plotting utilities; colors, fonts, plot styles.

Usage:

from llm_synthesis.utils.style_utils import get_cmap, get_palette, set_style

set_style("manuscript") # Sets font, font size, figure size, color palette etc.
palette = get_palette() # Get the palette for the current style
cmap = get_cmap() # Get the cmap for the current style

"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from cmcrameri import cm

cmap = cm.batlow
palette = cmap.colors
# convert to list
palette = list(palette)
palette.append("#D3D3D3")
palette.append("#A9A9A9")
palette.append("#808080")
sns.set_palette(palette)


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
        "font.family": "Latin Modern Sans",
        "mathtext.fontset": "dejavusans",
        "text.latex.preamble": (
            r"\usepackage{amsmath} \usepackage{amssymb} \usepackage{sfmath}"
        ),
        # Figure size
        "figure.figsize": (3.3, 2.5),
        "figure.dpi": 300,
        "figure.facecolor": "white",
        # Tick parameters
        "xtick.direction": "in",
        "ytick.direction": "in",
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
