"""Regenerate Fig2-style appendix figures/tables for the 'full' config
(58.3k procedures), matching the 'high_score'-config figures already
produced in Fig2_Dataset_Statistics.ipynb. Outputs saved with an
'_overall' suffix under examples/notebooks/figures/.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from datasets import load_dataset
from matplotlib.colors import LinearSegmentedColormap

from llm_synthesis.utils.style_utils import (
    BORDER_STYLE as _BORDER_STYLE,
)
from llm_synthesis.utils.style_utils import (
    get_palette,
    plot_category_bar,
    save_fig,
    set_style,
)

palette = get_palette()
set_style()

FIG_DIR = Path(__file__).parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

SOURCE_ORDER = ["arxiv", "chemrxiv", "omg24"]


def build_appendix_table(df, column):
    total = len(df)
    counts = df[column].value_counts()
    pct = (counts / total * 100).round(2)
    by_source = (
        df.groupby([column, "source"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=SOURCE_ORDER)
    )
    table = pd.DataFrame({"count": counts, "pct": pct}).join(by_source)
    return table.sort_values("count", ascending=False)


# --- load 'full' config (overall dataset, 58.3k procedures) ---
ds_full = load_dataset("LeMaterial/LeMat-Synth", "full")
_dfs = []
for split, d in ds_full.items():
    df = d.to_pandas()
    df["source"] = split
    _dfs.append(df)
df_overall = pd.concat(_dfs, ignore_index=True)
print(f"{len(df_overall)} materials (overall/full config)")

# --- category & method bar charts (linear + log-scale) ---
_BAR_CONFIGS = [
    dict(
        column="material_category",
        top_n=7,
        name="overall_material-category_top7",
    ),
    dict(
        column="material_category",
        top_n=7,
        stack_by="source",
        name="overall_material-category_top7_bysource",
    ),
    dict(
        column="material_category",
        top_n=None,
        name="overall_material-category_all",
    ),
    dict(
        column="synthesis_method",
        top_n=7,
        drop=["other"],
        name="overall_synthesis-method_top7",
    ),
    dict(
        column="synthesis_method",
        top_n=7,
        stack_by="material_category",
        drop=["other"],
        name="overall_synthesis-method_top7_bycategory",
    ),
    dict(
        column="synthesis_method",
        top_n=None,
        name="overall_synthesis-method_all",
    ),
]

for cfg in _BAR_CONFIGS:
    plot_category_bar(df_overall, fig_dir=FIG_DIR, **cfg)
    plt.close("all")
    plot_category_bar(
        df_overall,
        fig_dir=FIG_DIR,
        log_scale=True,
        **{**cfg, "name": cfg["name"] + "_log"},
    )
    plt.close("all")

# --- diversity: coverage grid + heatmaps ---
heat_cmap = LinearSegmentedColormap.from_list(
    "brand_sequential", [palette[6], palette[0], palette[2]]
)
df_clean = df_overall[
    (df_overall["material_category"] != "other")
    & (df_overall["synthesis_method"] != "other")
]
TOP_N_DIVERSITY = 7
top_cat = (
    df_clean["material_category"]
    .value_counts()
    .index[:TOP_N_DIVERSITY]
    .tolist()
)
top_method = (
    df_clean["synthesis_method"].value_counts().index[:TOP_N_DIVERSITY].tolist()
)

all_cat = df_clean["material_category"].value_counts().index.tolist()
all_method = df_clean["synthesis_method"].value_counts().index.tolist()
full_pivot = (
    df_clean.groupby(["material_category", "synthesis_method"])
    .size()
    .unstack(fill_value=0)
    .reindex(index=all_cat, columns=all_method)
)
with plt.rc_context(_BORDER_STYLE):
    fig, ax = plt.subplots(figsize=(10, 7))
    ys, xs, sizes = [], [], []
    for i, cat in enumerate(all_cat):
        for j, meth in enumerate(all_method):
            v = full_pivot.loc[cat, meth]
            if v > 0:
                ys.append(i)
                xs.append(j)
                sizes.append(v)
    import numpy as np

    sizes = np.array(sizes)
    ax.scatter(
        xs,
        ys,
        s=np.sqrt(sizes) * 3,
        color=palette[2],
        alpha=0.7,
        edgecolors="black",
        linewidths=0.3,
    )
    ax.set_xticks(range(len(all_method)))
    ax.set_xticklabels(all_method, rotation=60, ha="right")
    ax.set_yticks(range(len(all_cat)))
    ax.set_yticklabels(all_cat, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Synthesis Method")
    ax.set_ylabel("Material Category")
    plt.tight_layout()
    save_fig(fig, FIG_DIR, "overall_diversity-coverage-grid")
plt.close("all")

sub = df_clean[
    df_clean["material_category"].isin(top_cat)
    & df_clean["synthesis_method"].isin(top_method)
]
pivot = (
    sub.groupby(["material_category", "synthesis_method"])
    .size()
    .unstack(fill_value=0)
    .reindex(index=top_cat, columns=top_method)
)
with plt.rc_context(_BORDER_STYLE):
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        pivot,
        annot=True,
        fmt="d",
        cmap=heat_cmap,
        linewidths=0.5,
        cbar_kws={"label": "# Materials"},
        ax=ax,
    )
    ax.set_xlabel("Synthesis Method")
    ax.set_ylabel("Material Category")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    plt.tight_layout()
    save_fig(fig, FIG_DIR, "overall_diversity-heatmap-counts_top7")
plt.close("all")

pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100
with plt.rc_context(_BORDER_STYLE):
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        pivot_pct,
        annot=True,
        fmt=".0f",
        cmap=heat_cmap,
        linewidths=0.5,
        cbar_kws={"label": "% within category"},
        ax=ax,
    )
    ax.set_xlabel("Synthesis Method")
    ax.set_ylabel("Material Category")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    plt.tight_layout()
    save_fig(fig, FIG_DIR, "overall_diversity-heatmap-rowpct_top7")
plt.close("all")

from matplotlib.colors import LogNorm  # noqa: E402

with plt.rc_context(_BORDER_STYLE):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        full_pivot,
        mask=full_pivot == 0,
        norm=LogNorm(vmin=1, vmax=full_pivot.values.max()),
        cmap=heat_cmap,
        linewidths=0.2,
        cbar_kws={"label": "# Materials (log scale)"},
        ax=ax,
    )
    ax.set_xlabel("Synthesis Method")
    ax.set_ylabel("Material Category")
    plt.setp(ax.get_xticklabels(), rotation=60, ha="right", fontsize=7)
    plt.setp(ax.get_yticklabels(), fontsize=8)
    plt.tight_layout()
    save_fig(fig, FIG_DIR, "overall_diversity-heatmap-counts_all")
plt.close("all")

# --- appendix tables ---
build_appendix_table(df_overall, "material_category").to_csv(
    FIG_DIR / "overall_material-category_table-all.csv"
)
build_appendix_table(df_overall, "synthesis_method").to_csv(
    FIG_DIR / "overall_synthesis-method_table-all.csv"
)

print("Done. Overall-config figures/tables saved under", FIG_DIR)
