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

from llm_synthesis.utils.style_utils import get_palette, set_style

palette = get_palette()
set_style()

FIG_DIR = Path(__file__).parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

SOURCE_ORDER = ["arxiv", "chemrxiv", "omg24"]
_BORDER_STYLE = {
    "axes.spines.top": True,
    "axes.spines.right": True,
    "axes.linewidth": 1.4,
    "xtick.major.width": 1.2,
    "ytick.major.width": 1.2,
    "xtick.major.size": 4,
    "ytick.major.size": 4,
}


def save_fig(fig, name):
    fig.savefig(FIG_DIR / f"{name}.svg", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{name}.pdf", bbox_inches="tight")


def plot_category_bar(
    df, column, top_n=None, stack_by=None, drop=None, name=None
):
    if drop:
        df = df[~df[column].isin(drop)]

    order = df[column].value_counts().index.tolist()
    if top_n is not None:
        order = order[:top_n]

    with plt.rc_context(_BORDER_STYLE):
        fig, ax = plt.subplots(figsize=(6, 0.4 * len(order) + 1))

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
            ax.legend(
                title=stack_by.replace("_", " ").title(),
                bbox_to_anchor=(1.02, 1),
                loc="upper left",
                frameon=False,
            )
        else:
            counts = df[column].value_counts().reindex(order)
            bar_colors = [palette[i % len(palette)] for i in range(len(order))]
            bars = ax.barh(
                counts.index, counts.values, color=bar_colors, height=0.6
            )
            ax.bar_label(bars, fmt="%d", padding=3, fontsize=8)

        ax.set_xlabel("# Materials")
        ax.set_ylabel(column.replace("_", " ").title())
        ax.invert_yaxis()
        fig.tight_layout()
        if name:
            save_fig(fig, name)
    return fig, ax


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

# --- category & method bar charts ---
plot_category_bar(
    df_overall,
    "material_category",
    top_n=7,
    name="fig2_material_category_pub_overall",
)
plt.close("all")
plot_category_bar(
    df_overall,
    "material_category",
    top_n=7,
    stack_by="source",
    name="fig2_material_category_pub_by_source_overall",
)
plt.close("all")
plot_category_bar(
    df_overall,
    "material_category",
    top_n=None,
    name="fig2_material_category_appendix_overall",
)
plt.close("all")

plot_category_bar(
    df_overall,
    "synthesis_method",
    top_n=7,
    drop=["other"],
    name="fig2_synthesis_method_pub_overall",
)
plt.close("all")
plot_category_bar(
    df_overall,
    "synthesis_method",
    top_n=7,
    stack_by="material_category",
    drop=["other"],
    name="fig2_synthesis_method_pub_by_category_overall",
)
plt.close("all")
plot_category_bar(
    df_overall,
    "synthesis_method",
    top_n=None,
    name="fig2_synthesis_method_appendix_overall",
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
    save_fig(fig, "fig2_diversity_coverage_grid_overall")
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
    save_fig(fig, "fig2_diversity_heatmap_counts_overall")
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
    save_fig(fig, "fig2_diversity_heatmap_rowpct_overall")
plt.close("all")

# --- appendix tables ---
build_appendix_table(df_overall, "material_category").to_csv(
    FIG_DIR / "table_material_category_appendix_overall.csv"
)
build_appendix_table(df_overall, "synthesis_method").to_csv(
    FIG_DIR / "table_synthesis_method_appendix_overall.csv"
)

print("Done. Overall-config figures/tables saved under", FIG_DIR)
