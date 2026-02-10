"""
Map of Science for Catalysis — 5 Publication-Quality Example Figures
====================================================================
Synthetic (made-up) data to illustrate visualisation concepts for
comparing NH3 decomposition / synthesis catalysts across papers.

Run:  python catalysis_map_figures.py
Outputs 5 PDF + PNG figures in the current directory.
"""

import itertools

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D

# ── Global style ──────────────────────────────────────────────────────────
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 11,
        "axes.linewidth": 1.2,
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "figure.dpi": 180,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)

# A muted, publication-friendly palette
PAL = {
    "Ru": "#2176AE",  # blue
    "Ni": "#E07A5F",  # terracotta
    "Co": "#81B29A",  # sage green
    "Fe": "#F2CC8F",  # sand
    "NiCo": "#3D405B",  # charcoal
    "FeCo": "#7B2D8E",  # purple
    "Pt": "#D4A017",  # gold
}

SUPPORT_MARKERS = {
    "SiO₂": "o",
    "CeO₂": "s",
    "MgO": "D",
    "Al₂O₃": "^",
    "CaO": "P",
    "MWCNT": "*",
}

# ══════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Cross-Paper NH3 Conversion Landscape
# ══════════════════════════════════════════════════════════════════════════


def sigmoid_curve(temp, temp50, k, y_max=100):
    """Generate a sigmoid-shaped conversion curve."""
    return y_max / (1 + np.exp(-k * (temp - temp50)))


def make_fig1():
    np.random.seed(42)
    temp = np.linspace(300, 700, 80)

    catalysts = [
        # (label, metal_key, support, temp50, steepness, max_conv, paper)
        ("Ru/MgO(111)", "Ru", "MgO", 380, 0.035, 99, "Fang 2023"),
        ("Ru/MgO(100)", "Ru", "MgO", 430, 0.030, 95, "Fang 2023"),
        ("Ru/MgO(110)", "Ru", "MgO", 410, 0.032, 97, "Fang 2023"),
        ("3%Ru10%K/CaO", "Ru", "CaO", 400, 0.028, 98, "Sayas 2020"),
        ("5%Ru10%K/MgO", "Ru", "MgO", 420, 0.026, 96, "Sayas 2020"),
        ("Ni₇Co₃/SiO₂", "NiCo", "SiO₂", 520, 0.022, 88, "Wu 2020"),
        ("Ni₅Co₅/SiO₂", "NiCo", "SiO₂", 510, 0.024, 91, "Wu 2020"),
        ("Ni₅Co₅/SiO₂-K", "NiCo", "SiO₂", 480, 0.026, 94, "Wu 2020"),
        ("Ni/SiO₂", "Ni", "SiO₂", 560, 0.020, 82, "Wu 2020"),
        ("Co/SiO₂", "Co", "SiO₂", 580, 0.018, 75, "Wu 2020"),
        ("FeCo/CeO₂-S", "FeCo", "CeO₂", 500, 0.023, 86, "Gao 2023"),
        ("Ni/Al₂O₃", "Ni", "Al₂O₃", 540, 0.021, 84, "Do 2024"),
        ("Ni/CeO₂", "Ni", "CeO₂", 530, 0.022, 87, "Lucentini 2019"),
        ("Co/CeO₂-S", "Co", "CeO₂", 570, 0.019, 78, "Gao 2023"),
    ]

    fig, ax = plt.subplots(figsize=(10, 6))

    for label, mk, sup, temp50, k, ymax, paper in catalysts:
        y = sigmoid_curve(temp, temp50, k, ymax) + np.random.normal(
            0, 0.8, len(temp)
        )
        y = np.clip(y, 0, 100)
        color = PAL[mk]
        marker = SUPPORT_MARKERS.get(sup, "o")
        ax.plot(temp, y, color=color, alpha=0.75, linewidth=1.6)
        # mark every 15th point
        ax.scatter(
            temp[::15],
            y[::15],
            color=color,
            marker=marker,
            s=30,
            zorder=3,
            edgecolors="white",
            linewidth=0.4,
        )

    # ── Reference lines ──
    ax.axhline(90, color="grey", ls="--", lw=0.8, alpha=0.5)
    ax.text(305, 91, "90 % target", fontsize=8, color="grey")

    # ── Legend: metals ──
    metal_handles = [
        Line2D([0], [0], color=PAL[m], lw=2.5, label=m)
        for m in ["Ru", "Ni", "Co", "NiCo", "FeCo"]
    ]
    leg1 = ax.legend(
        handles=metal_handles,
        title="Active Metal",
        loc="lower right",
        frameon=True,
        framealpha=0.9,
        edgecolor="grey",
    )
    ax.add_artist(leg1)

    # ── Legend: supports ──
    sup_handles = [
        Line2D(
            [0],
            [0],
            marker=SUPPORT_MARKERS[s],
            color="grey",
            lw=0,
            markersize=7,
            label=s,
        )
        for s in ["SiO₂", "CeO₂", "MgO", "CaO", "Al₂O₃"]
    ]
    ax.legend(
        handles=sup_handles,
        title="Support",
        loc="center right",
        frameon=True,
        framealpha=0.9,
        edgecolor="grey",
        bbox_to_anchor=(1.0, 0.55),
    )

    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("NH₃ Conversion (%)")
    ax.set_title(
        "Figure 1 · Cross-Paper NH₃ Conversion Landscape", fontweight="bold"
    )
    ax.set_xlim(300, 700)
    ax.set_ylim(0, 105)
    ax.grid(axis="both", alpha=0.2)

    fig.tight_layout()
    fig.savefig("fig1_conversion_landscape.png")
    fig.savefig("fig1_conversion_landscape.pdf")
    plt.close(fig)
    print("✓ Figure 1 saved")


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Metal x Support Heatmap ("Catalyst Periodic Table")
# ══════════════════════════════════════════════════════════════════════════


def make_fig2():
    metals = ["Ru", "Ni", "Co", "Fe", "Ni-Co", "Fe-Co", "Fe-Ni", "Pt"]
    supports = ["MgO", "CaO", "CeO₂", "SiO₂", "Al₂O₃", "MWCNT", "TiO₂"]

    np.random.seed(7)
    data = np.full((len(metals), len(supports)), np.nan)

    known = {
        (0, 0): 99,
        (0, 1): 98,
        (0, 2): 82,
        (0, 5): 90,  # Ru
        (1, 2): 87,
        (1, 3): 82,
        (1, 4): 84,  # Ni
        (2, 2): 78,
        (2, 3): 75,  # Co
        (3, 2): 70,  # Fe
        (4, 3): 91,
        (4, 2): 80,  # Ni-Co
        (5, 2): 86,  # Fe-Co
        (6, 2): 83,  # Fe-Ni
        (7, 6): 65,  # Pt
    }
    for (r, c), v in known.items():
        data[r, c] = v

    fig, ax = plt.subplots(figsize=(9, 5.5))

    cmap_heat = plt.cm.YlOrRd.copy()
    cmap_heat.set_bad(color="#f0f0f0")

    im = ax.imshow(data, cmap=cmap_heat, vmin=50, vmax=100, aspect="auto")

    # ── Annotate cells ──
    for i in range(len(metals)):
        for j in range(len(supports)):
            val = data[i, j]
            if np.isnan(val):
                ax.text(
                    j,
                    i,
                    "—",
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="#bbbbbb",
                )
            else:
                txt_color = "white" if val > 85 else "black"
                ax.text(
                    j,
                    i,
                    f"{val:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=11,
                    fontweight="bold",
                    color=txt_color,
                )

    ax.set_xticks(range(len(supports)))
    ax.set_xticklabels(supports)
    ax.set_yticks(range(len(metals)))
    ax.set_yticklabels(metals)
    ax.set_xlabel("Support Material")
    ax.set_ylabel("Active Metal / Alloy")
    ax.set_title(
        "Figure 2 · Best NH₃ Conversion at 500 °C by Metal-Support Combination",
        fontweight="bold",
    )

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Max NH₃ Conversion (%)")

    # ── Annotate unexplored ──
    ax.annotate(
        "grey = unexplored\ncombination",
        xy=(5.5, 6.5),
        fontsize=8,
        color="grey",
        ha="center",
        style="italic",
    )

    fig.tight_layout()
    fig.savefig("fig2_metal_support_heatmap.png")
    fig.savefig("fig2_metal_support_heatmap.pdf")
    plt.close(fig)
    print("✓ Figure 2 saved")


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Synthesis Method Network Graph
# ══════════════════════════════════════════════════════════════════════════


def make_fig3():
    # Synthesis action sequences extracted across papers (representative)
    sequences = [
        # Wu 2020 - wet impregnation
        [
            "dissolve",
            "disperse",
            "add\nprecursor",
            "sonicate",
            "stir",
            "evaporate",
            "dry",
            "sieve",
            "reduce",
        ],
        [
            "dissolve",
            "disperse",
            "add\nprecursor",
            "sonicate",
            "stir",
            "evaporate",
            "dry",
            "sieve",
            "reduce",
        ],
        # Fang 2023 - hydrothermal + impregnation
        [
            "dissolve",
            "stir",
            "precipitate",
            "hydrothermal",
            "wash",
            "dry",
            "calcine",
            "impregnate",
            "stir",
            "dry",
            "heat\ntreat",
        ],
        # Sayas 2020
        [
            "dissolve",
            "impregnate",
            "dry",
            "calcine",
            "add\npromoter",
            "dry",
            "reduce",
        ],
        # Do 2024
        [
            "dissolve",
            "add\nprecursor",
            "stir",
            "precipitate",
            "wash",
            "dry",
            "calcine",
            "reduce",
        ],
        # Gao 2023 - coprecipitation
        [
            "dissolve",
            "add\nprecursor",
            "coprecipitate",
            "wash",
            "dry",
            "calcine",
            "reduce",
        ],
        # Lucentini 2019
        ["dissolve", "impregnate", "stir", "dry", "calcine", "reduce"],
        # Hu 2024
        [
            "dissolve",
            "add\nprecursor",
            "stir",
            "precipitate",
            "age",
            "wash",
            "dry",
            "calcine",
            "impregnate",
            "dry",
            "reduce",
        ],
        # Maleki 2024
        ["dissolve", "coprecipitate", "wash", "dry", "calcine", "reduce"],
    ]
    node_counts = {}
    edge_counts = {}
    graph = nx.DiGraph()

    for seq in sequences:
        for node in seq:
            node_counts[node] = node_counts.get(node, 0) + 1
        for a, b in itertools.pairwise(seq):
            edge_counts[(a, b)] = edge_counts.get((a, b), 0) + 1

    for node, count in node_counts.items():
        graph.add_node(node, weight=count)
    for (a, b), count in edge_counts.items():
        graph.add_edge(a, b, weight=count)

    fig, ax = plt.subplots(figsize=(12, 8))

    pos = nx.spring_layout(graph, seed=42, k=2.2, iterations=80)

    # ── Node sizes & colors ──
    node_sizes = [node_counts[n] * 220 + 200 for n in graph.nodes()]
    # node_colors_vals = [node_counts[n] for n in graph.nodes()]
    norm = Normalize(vmin=1, vmax=max(node_counts.values()))
    cmap_nodes = plt.cm.Blues

    node_colors = [cmap_nodes(norm(node_counts[n])) for n in graph.nodes()]

    # ── Edge widths ──
    edge_widths = [edge_counts[(u, v)] * 1.5 + 0.3 for u, v in graph.edges()]
    # edge_colors = [
    #     f"#{int(180 - edge_counts[(u,v)]*15):02x}{int(180 - edge_counts[(u,v)]*15):02x}{int(180 - edge_counts[(u,v)]*15):02x}" # noqa: E501
    #     for u, v in G.edges()
    # ]

    nx.draw_networkx_edges(
        graph,
        pos,
        ax=ax,
        width=edge_widths,
        edge_color="#999999",
        alpha=0.6,
        arrows=True,
        arrowsize=18,
        connectionstyle="arc3,rad=0.1",
        min_source_margin=18,
        min_target_margin=18,
    )

    nx.draw_networkx_nodes(
        graph,
        pos,
        ax=ax,
        node_size=node_sizes,
        node_color=node_colors,
        edgecolors="white",
        linewidths=1.5,
    )

    nx.draw_networkx_labels(graph, pos, ax=ax, font_size=8, font_weight="bold")

    # ── Edge labels for frequent transitions ──
    edge_labels = {
        (u, v): str(w) for (u, v), w in edge_counts.items() if w >= 3
    }
    nx.draw_networkx_edge_labels(
        graph,
        pos,
        edge_labels=edge_labels,
        ax=ax,
        font_size=7,
        font_color="#E07A5F",
        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8),
    )

    sm = ScalarMappable(cmap=cmap_nodes, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label("Frequency across papers")

    ax.set_title(
        "Figure 3 · Synthesis Action Network — Sequential Steps across 12 Papers",  # noqa: E501
        fontweight="bold",
        fontsize=13,
    )
    ax.axis("off")

    fig.tight_layout()
    fig.savefig("fig3_synthesis_network.png")
    fig.savefig("fig3_synthesis_network.pdf")
    plt.close(fig)
    print("✓ Figure 3 saved")


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Synthesis-Performance Radar Charts (Top 6 catalysts)
# ══════════════════════════════════════════════════════════════════════════


def make_fig4():
    categories = [
        "NH₃ Conv.\n@ 500°C (%)",
        "Metal Loading\n(wt%)",
        "Calcination T\n(°C / 100)",
        "Reduction T\n(°C / 100)",
        "Synthesis\nSteps (#)",
        "T₅₀\n(°C / 100)",
    ]

    # Normalised data (all scaled 0-1 for radar)
    catalysts_data = {
        "Ru/MgO(111)\nFang 2023": [0.99, 0.15, 5.0, 3.0, 12, 3.8],
        "3%Ru10%K/CaO\nSayas 2020": [0.98, 0.30, 5.0, 4.5, 7, 4.0],
        "Ni₅Co₅/SiO₂-K\nWu 2020": [0.94, 1.00, 0.0, 5.5, 8, 4.8],
        "Ni₅Co₅/SiO₂\nWu 2020": [0.91, 1.00, 0.0, 5.5, 8, 5.1],
        "FeCo/CeO₂-S\nGao 2023": [0.86, 0.50, 4.0, 5.0, 7, 5.0],
        "Ni/Al₂O₃\nDo 2024": [0.84, 0.80, 4.5, 5.5, 8, 5.4],
    }

    # Normalise each axis to [0, 1]
    raw = np.array(list(catalysts_data.values()))
    mins = raw.min(axis=0)
    maxs = raw.max(axis=0)
    ranges = maxs - mins
    ranges[ranges == 0] = 1

    fig, axes = plt.subplots(2, 3, figsize=(14, 9), subplot_kw=dict(polar=True))
    axes = axes.flatten()

    colors = ["#2176AE", "#1B998B", "#3D405B", "#E07A5F", "#7B2D8E", "#D4A017"]

    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    for idx, (name, values) in enumerate(catalysts_data.items()):
        ax = axes[idx]
        # Normalise
        norm_vals = [(v - mi) / r for v, mi, r in zip(values, mins, ranges)]
        norm_vals += norm_vals[:1]

        ax.fill(angles, norm_vals, color=colors[idx], alpha=0.2)
        ax.plot(angles, norm_vals, color=colors[idx], linewidth=2)
        ax.scatter(
            angles[:-1], norm_vals[:-1], color=colors[idx], s=40, zorder=5
        )

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=7)
        ax.set_ylim(0, 1.1)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["", "", "", ""], fontsize=6)
        ax.set_title(
            name, fontsize=9, fontweight="bold", pad=18, color=colors[idx]
        )
        ax.grid(alpha=0.3)

    fig.suptitle(
        "Figure 4 · Synthesis Parameter-Performance Radar Charts\nfor Top NH₃ Decomposition Catalysts",  # noqa: E501
        fontweight="bold",
        fontsize=14,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig("fig4_radar_charts.png")
    fig.savefig("fig4_radar_charts.pdf")
    plt.close(fig)
    print("✓ Figure 4 saved")


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 5 — Combined: Promoter Effect + Synthesis Conditions Scatter
# ══════════════════════════════════════════════════════════════════════════


def make_fig5():
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(14, 5.5), gridspec_kw={"width_ratios": [1, 1.2]}
    )

    # ── Panel A: Promoter Effect Δ-conversion ──
    promoters = {
        "K → Ni₅Co₅/SiO₂": (91, 94, "K"),
        "K → 3%Ru/CaO": (78, 98, "K"),
        "La → Ni/Al₂O₃": (84, 90, "La"),
        "Ce → Ni/Al₂O₃": (84, 92, "Ce"),
        "Nd → Ni/Al₂O₃": (84, 88, "Nd"),
        "Sm → Ni/Al₂O₃": (84, 86, "Sm"),
        "Ca → Ru/MgO": (95, 97, "Ca"),
    }

    prom_colors = {
        "K": "#2176AE",
        "La": "#E07A5F",
        "Ce": "#81B29A",
        "Nd": "#F2CC8F",
        "Sm": "#3D405B",
        "Ca": "#D4A017",
    }

    labels = list(promoters.keys())
    base_vals = [v[0] for v in promoters.values()]
    promoted_vals = [v[1] for v in promoters.values()]
    deltas = [p - b for b, p in zip(base_vals, promoted_vals)]
    bar_colors = [prom_colors[v[2]] for v in promoters.values()]

    y_pos = np.arange(len(labels))

    # Base bars (light)
    ax1.barh(
        y_pos,
        base_vals,
        height=0.5,
        color="#dddddd",
        edgecolor="white",
        label="Unpromoted",
    )
    # Promoted extension
    ax1.barh(
        y_pos,
        deltas,
        left=base_vals,
        height=0.5,
        color=bar_colors,
        edgecolor="white",
        alpha=0.85,
    )

    for i, (b, p, d) in enumerate(zip(base_vals, promoted_vals, deltas)):
        ax1.text(
            p + 0.5,
            i,
            f"+{d}%",
            va="center",
            fontsize=9,
            fontweight="bold",
            color=bar_colors[i],
        )

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(labels, fontsize=9)
    ax1.set_xlabel("NH₃ Conversion at 500 °C (%)")
    ax1.set_xlim(60, 105)
    ax1.set_title("(a) Promoter Effect on NH₃ Conversion", fontweight="bold")
    ax1.grid(axis="x", alpha=0.2)
    ax1.legend(
        ["Unpromoted baseline", "Δ from promoter"],
        loc="lower right",
        fontsize=8,
    )

    # ── Panel B: Synthesis Conditions → Performance Bubble Chart ──
    np.random.seed(123)
    n = 25
    calc_temp = np.random.uniform(350, 700, n)  # calcination T
    red_temp = np.random.uniform(400, 600, n)  # reduction T
    conversion = (
        30
        + 50 * np.exp(-0.003 * (red_temp - 450) ** 2 / 50)
        + np.random.normal(0, 5, n)
    )
    conversion = np.clip(conversion, 20, 100)
    metal_load = np.random.uniform(1, 15, n)  # wt%
    # n_steps = np.random.randint(5, 14, n)

    # Color by conversion, size by metal loading
    scatter = ax2.scatter(
        calc_temp,
        red_temp,
        c=conversion,
        cmap="RdYlGn",
        s=metal_load * 25 + 30,
        edgecolors="white",
        linewidth=0.6,
        vmin=40,
        vmax=100,
        alpha=0.85,
        zorder=3,
    )

    cbar = fig.colorbar(scatter, ax=ax2, shrink=0.8, pad=0.02)
    cbar.set_label("NH₃ Conversion (%)")

    # Size legend
    for ml, lab in [(2, "2 wt%"), (8, "8 wt%"), (15, "15 wt%")]:
        ax2.scatter(
            [],
            [],
            s=ml * 25 + 30,
            c="grey",
            alpha=0.5,
            edgecolors="white",
            label=lab,
        )
    ax2.legend(
        title="Metal Loading",
        loc="upper left",
        fontsize=8,
        title_fontsize=9,
        framealpha=0.9,
    )

    ax2.set_xlabel("Calcination Temperature (°C)")
    ax2.set_ylabel("Reduction Temperature (°C)")
    ax2.set_title("(b) Synthesis Conditions → Performance", fontweight="bold")
    ax2.grid(alpha=0.2)

    fig.suptitle(
        "Figure 5 · Promoter Effects & Synthesis-Performance Correlations",
        fontweight="bold",
        fontsize=14,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig("fig5_promoter_and_conditions.png")
    fig.savefig("fig5_promoter_and_conditions.pdf")
    plt.close(fig)
    print("✓ Figure 5 saved")


# ══════════════════════════════════════════════════════════════════════════
# RUN ALL
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Generating 5 publication figures with synthetic data...\n")
    make_fig1()
    make_fig2()
    make_fig3()
    make_fig4()
    make_fig5()
    print("\n✅ All 5 figures saved as PNG + PDF in current directory.")
