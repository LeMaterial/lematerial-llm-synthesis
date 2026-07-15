"""
Map of Science — Publication-Quality Figures from Extracted Data
================================================================
Loads LLM-extracted performance + synthesis data from a results folder
and produces 8 figures (PNG + PDF) + companion CSV.

Fully generic: works with any set of papers and any performance metric.

Run:  uv run python catalysis_map_figures.py /path/to/results_folder
      uv run python catalysis_map_figures.py /path/to/results_folder --debug
      uv run python catalysis_map_figures.py /path/to/results_folder --use-llm
"""

import argparse
import itertools
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D

# ── Project style system ─────────────────────────────────────────────────
SRC_DIR = str(Path(__file__).resolve().parents[3] / "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from llm_synthesis.utils.style_utils import get_palette, set_style  # noqa: E402

set_style("presentation")
PAL = get_palette()

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "axes.grid": False,
    }
)

# ── Paths (DATA_DIR set from CLI, OUT_DIR defaults to figure_visualisation/) ──
DATA_DIR = None  # set in __main__ from CLI argument; type: Path
OUT_DIR = Path(".")  # overridden in __main__ from --out-dir or data_dir

# ── Skip / filter constants ─────────────────────────────────────────────
SKIP_FILES = {
    "linking_summary_human.json",
    "linking_summary_llm.json",
    "performance_mappings.json",
    "batch_summary.json",
    "summary.json",
}

GENERIC_SERIES = {
    "red triangles",
    "circle markers",
    "triangle markers",
    "square markers",
    "green squares",
    "blue circles",
    "black diamonds",
    "lower_performing_curve",
    "middle_performing_curve",
    "top_performing_curve",
    "catalyst",
    "plasma+catalyst",
    "blank",
    "plasma",
    "plasma on (9095 v)",
    "thermodynamic equilibrium",
}

# ── Configurable metric settings (set via CLI or auto-detected) ─────────
# These are overridden in __main__ based on
# --y-label / --ref-temp / auto-detection.
Y_LABEL = "Conversion (%)"  # y-axis label for figures
Y_KEYWORDS = ["conversion"]  # keywords to match y_axis_label in plot_data
REF_TEMP = 500.0  # reference temperature for interpolation
METRIC_NAME = "conversion"  # short name for filenames and column headers

# ── Known elements for material parsing ─────────────────────────────────
# Active metals commonly found in materials science
KNOWN_METALS = {
    "Ru",
    "Ni",
    "Co",
    "Fe",
    "Mo",
    "Pt",
    "Mn",
    "Cu",
    "Pd",
    "W",
    "Cr",
    "Re",
    "Ir",
    "Rh",
    "Os",
    "Au",
    "Ag",
    "V",
    "Nb",
    "Ta",
    "Ti",
    "Zr",
    "Hf",
}

# Elements typically used as promoters (alkali, alkaline earth, rare earth)
KNOWN_PROMOTERS = {
    "K",
    "Na",
    "Ca",
    "Ba",
    "Sr",
    "Cs",
    "Li",
    "La",
    "Ce",
    "Nd",
    "Sm",
    "Gd",
    "Pr",
    "Y",
}

# Known perovskite formulas
PEROVSKITES = {
    "BaTiO3",
    "SrTiO3",
    "CaTiO3",
    "BaZrO3",
    "SrZrO3",
    "CaZrO3",
    "BaMnO3",
    "CaMnO3",
    "SrMnO3",
    "GdAlO3",
    "KNbO3",
    "LaAlO3",
    "NaNbO3",
    "SmAlO3",
    "LaFeO3",
    "LaCoO3",
    "LaMnO3",
}

# Generic ABO3 perovskite detector
_PEROVSKITE_RE = re.compile(r"^[A-Z][a-z]?[A-Z][a-z]?O3$")

# ── Preferred color / marker assignments ────────────────────────────────
# These are used first; overflow metals/supports get auto-assigned.
# Preferred display order per category (colors are assigned from PAL by
# position in this order, so legend order and hue assignment stay stable
# across VLM runs even as new metals/supports/strategies show up).
_PREFERRED_METAL_ORDER = [
    "Ru",
    "Ni",
    "Co",
    "Fe",
    "Mo",
    "Pt",
    "Pd",
    "W",
    "Ir",
    "NiCo",
    "FeCo",
    "FeNi",
    "CoNi",
    "RuFe",
    "CoMo",
    "NiRu",
    "RuNi",
]
_PREFERRED_SUPPORT_ORDER = [
    "SiO2",
    "CeO2",
    "MgO",
    "Al2O3",
    "CaO",
    "MCM-41",
    "CNTs",
    "MgAl2O4",
    "TiCSiC",
    "BN",
    "Mo2N",
    "perovskite",
    "Y2O3",
    "ZrO2",
    "TiO2",
    "La2O3",
]

_auto_metal_colors: dict = {}
_auto_support_colors: dict = {}


def _palette_color(category: str, preferred_order: list, cache: dict) -> str:
    """Assign a PAL color to `category` by its position in `preferred_order`,
    falling back to append-and-cycle for anything not in that list."""
    if category == "Other":
        return "#000000"
    if category in preferred_order:
        idx = preferred_order.index(category)
        return PAL[idx % len(PAL)]
    if category not in cache:
        idx = (len(preferred_order) + len(cache)) % len(PAL)
        cache[category] = PAL[idx]
    return cache[category]


def get_metal_color(metal: str) -> str:
    """Return a PAL color for the given metal, auto-assigning if unknown."""
    return _palette_color(metal, _PREFERRED_METAL_ORDER, _auto_metal_colors)


def get_support_color(support: str) -> str:
    """Return a PAL color for the given support, auto-assigning if unknown."""
    return _palette_color(
        support, _PREFERRED_SUPPORT_ORDER, _auto_support_colors
    )


# ── Regex patterns ──────────────────────────────────────────────────────
VOLTAGE_RE = re.compile(
    r"\((\d+)\s*V\)", re.IGNORECASE
)  # detect voltage tags in series names
LOADING_RE = re.compile(r"(\d+\.?\d*)\s*(?:wt\.?%|wtpct|pct|%)")

# ── Strategy classification ─────────────────────────────────────────────
STRATEGY_ORDER = [
    "Impregnation",
    "Co-precipitation",
    "Sol-gel",
    "Hydrothermal",
    "Solid-state",
    "Combustion",
    "Oxide-only",
    "Other",
]

_auto_strategy_colors: dict = {}


def get_strategy_color(strategy: str) -> str:
    """Return a PAL color for the given synthesis strategy."""
    return _palette_color(strategy, STRATEGY_ORDER, _auto_strategy_colors)


# ══════════════════════════════════════════════════════════════════════════
# SECTION 1 — Data Loading & Filtering
# ══════════════════════════════════════════════════════════════════════════


def is_target_metric(label):
    """Check if a y-axis label matches the configured performance metric."""
    if not label:
        return False
    lo = label.lower().replace("₃", "3").replace("₂", "2")
    return any(kw in lo for kw in Y_KEYWORDS)


def interpolate_at_temp(coordinates, target_temp=500.0):
    """Linearly interpolate conversion at target_temp from coordinate pairs."""
    coords = np.array(coordinates, dtype=float)
    if len(coords) < 2:
        return np.nan
    temps, convs = coords[:, 0], coords[:, 1]
    if target_temp < temps.min() or target_temp > temps.max():
        return np.nan
    return float(np.interp(target_temp, temps, convs))


def normalize_series_name(name):
    """Normalize unicode subscripts/superscripts for dedup comparison."""
    if not name:
        return ""
    subs = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
    return name.translate(subs).strip().lower()


def load_all_data(
    skip_dirs=frozenset(),
    material_cache=None,
    DATA_DIR=DATA_DIR,  # noqa: N803
):
    """Walk all paper directories, load JSONs, return (df_curves, df_synthesis).

    If material_cache is provided (dict mapping material_name →
    {metal, support, loading}),
    uses cached LLM-parsed results instead of regex parsing.
    """
    curves_rows = []
    synth_rows = []

    for paper_dir_name in sorted(os.listdir(DATA_DIR)):
        paper_path = DATA_DIR / paper_dir_name
        if not paper_path.is_dir():
            continue
        if paper_dir_name in skip_dirs:
            continue

        for fname in sorted(os.listdir(paper_path)):
            if fname in SKIP_FILES or not fname.endswith(".json"):
                continue

            fpath = paper_path / fname
            try:
                with open(fpath, encoding="utf-8") as f:
                    d = json.load(f)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue

            mat_name = d.get("material", fname.replace(".json", ""))

            # ── Synthesis data (all materials) ──
            synth = d.get("synthesis", {}) or {}
            steps = synth.get("steps", []) or []
            actions = [s.get("action", "") for s in steps if s.get("action")]

            calc_t = None
            red_t = None
            for step in steps:
                cond = step.get("conditions") or {}
                temp = cond.get("temperature")
                action = step.get("action", "")
                if temp is not None:
                    if action == "calcine":
                        calc_t = max(calc_t or 0, temp)
                    elif action == "reduce":
                        red_t = max(red_t or 0, temp)

            if material_cache and mat_name in material_cache:
                cached = material_cache[mat_name]
                metal = cached.get("metal") or "Other"
                support = cached.get("support") or "Other"
                loading = cached.get("loading")
                if loading is None:
                    loading = np.nan
            else:
                metal, support, loading = parse_material_name(mat_name)
            synth_method = synth.get("synthesis_method", "")
            strategy = classify_synthesis_strategy(synth_method, actions, red_t)

            synth_rows.append(
                {
                    "paper_dir": paper_dir_name,
                    "material_name": mat_name,
                    "actions": actions,
                    "n_steps": len(steps),
                    "calcination_T": calc_t,
                    "reduction_T": red_t,
                    "metal": metal,
                    "support": support,
                    "metal_loading_pct": loading,
                    "strategy": strategy,
                }
            )

            # ── Performance data ──
            perf = d.get("performance")
            if not perf:
                continue

            plot_data_list = perf.get("plot_data", []) or []
            if not plot_data_list:
                continue

            series_groups = defaultdict(list)
            for pd_entry in plot_data_list:
                sname = pd_entry.get("series_name", "")
                ylabel = pd_entry.get("y_axis_label", "")
                coords = pd_entry.get("coordinates", [])

                if not is_target_metric(ylabel):
                    continue
                if normalize_series_name(sname) in GENERIC_SERIES:
                    continue
                if not coords or len(coords) < 2:
                    continue

                key = normalize_series_name(sname)
                series_groups[key].append((sname, coords, pd_entry))

            # Dedup: keep entry with most coordinate points per group
            for key, entries in series_groups.items():
                entries.sort(key=lambda x: len(x[1]), reverse=True)
                sname, coords, pd_entry = entries[0]

                is_plasma = False
                voltage = None
                vm = VOLTAGE_RE.search(sname)
                if vm:
                    is_plasma = True
                    voltage = vm.group(1) + " V"

                conv_500 = interpolate_at_temp(coords, REF_TEMP)

                curves_rows.append(
                    {
                        "paper_dir": paper_dir_name,
                        "material_name": mat_name,
                        "series_name": sname,
                        "coordinates": coords,
                        "metal": metal,
                        "support": support,
                        "metal_loading_pct": loading,
                        "is_plasma": is_plasma,
                        "voltage": voltage,
                        "conv_at_500": conv_500,
                        "strategy": strategy,
                    }
                )

    df_curves = pd.DataFrame(curves_rows)
    df_synthesis = pd.DataFrame(synth_rows)
    return df_curves, df_synthesis


# ══════════════════════════════════════════════════════════════════════════
# SECTION 2 — Material Name Parsing (generic, no paper-specific workarounds)
# ══════════════════════════════════════════════════════════════════════════


def parse_material_name(name):
    """Parse a catalyst name into (metal_category, support, loading_pct).

    Returns (str or None, str or None, float or NaN).
    """
    if not name:
        return None, None, np.nan

    # Extract loading %
    loading = np.nan
    m = LOADING_RE.search(name)
    if m:
        loading = float(m.group(1))

    # Normalize unicode subscripts for parsing
    subs = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
    name_norm = name.translate(subs)

    # Strip parenthetical notes BEFORE slash-splitting
    # to avoid "/" inside parens
    name_no_parens = re.sub(r"\s*\(.*\)$", "", name_norm).strip()

    # Strip loading prefix (e.g. "10 wt% ", "5.0 wt.% ", "3pct")
    name_clean = re.sub(
        r"^\d+\.?\d*\s*(?:wt\.?%?|wtpct|pct|%)\s*", "", name_no_parens
    ).strip()

    # Handle "10Ni/Al2O3" pattern — digits directly before metal, no % sign
    num_metal = re.match(r"^(\d+\.?\d*)([A-Z][a-z]?)(/.*)", name_clean)
    if num_metal and num_metal.group(2) in KNOWN_METALS:
        if np.isnan(loading):
            loading = float(num_metal.group(1))
        name_clean = num_metal.group(2) + num_metal.group(3)

    # Try splitting on "/" to get metal_part / support_part
    metal_part = None
    support_part = None

    if "/" in name_clean:
        slash_parts = name_clean.split("/")
        if len(slash_parts) == 3:
            # Check: is part[0] a promoter like "5%La"?
            p0 = re.sub(
                r"^\d+\.?\d*\s*(?:wt\.?%?|%)\s*", "", slash_parts[0].strip()
            )
            if p0 in KNOWN_PROMOTERS:
                # Promoter/Metal/Support: "5%La/Ni/Al2O3"
                metal_part = slash_parts[1].strip()
                support_part = slash_parts[2].strip()
            elif not any(p0.startswith(m) for m in KNOWN_METALS):
                # Metal/Oxide-additive/Support: "Ru/CeO2/MgAl2O4"
                metal_part = slash_parts[0].strip()
                support_part = slash_parts[2].strip()
            else:
                metal_part = slash_parts[0].strip()
                support_part = slash_parts[-1].strip()
        elif len(slash_parts) == 2:
            metal_part = slash_parts[0].strip()
            support_part = slash_parts[1].strip()
        else:
            metal_part = slash_parts[0].strip()
            support_part = slash_parts[-1].strip()
    elif name_clean in KNOWN_METALS:
        # Bare metal (e.g., "Fe")
        return name_clean, None, loading
    elif _looks_like_support(name_clean):
        # Bare support
        return None, _normalize_support(name_clean), loading
    elif "-" in name_clean:
        # Metal-Support pattern: "Fe-Al2O3", "Fe-CeO2"
        parts = name_clean.split("-", 1)
        lhs = parts[0].strip()
        rhs = parts[1].strip()
        if lhs in KNOWN_METALS:
            metal_part = lhs
            support_part = rhs
        else:
            return "Other", "Other", loading
    else:
        # Try generic patterns before giving up

        # Binary compound: Metal_xN_y or Metal_xC_y (nitrides, carbides)
        binary_match = re.match(r"^([A-Z][a-z]?)\d*[NC]\d*$", name_clean)
        if binary_match:
            metal_sym = binary_match.group(1)
            return (
                (metal_sym if metal_sym in KNOWN_METALS else "Other"),
                name_clean,
                loading,
            )

        # Spinel-type: MetalAl2O4, MetalFe2O4, etc.
        spinel_match = re.match(
            r"^([A-Z][a-z]?)([A-Z][a-z]?)(\d+)O(\d+)$", name_clean
        )
        if spinel_match:
            sm = spinel_match.group(1)
            base_metal = spinel_match.group(2)
            if sm in KNOWN_METALS:
                # Map spinel to parent oxide: Al2O4→Al2O3, Fe2O4→Fe2O3
                spinel_oxide_map = {
                    "Al2O4": "Al2O3",
                    "Fe2O4": "Fe2O3",
                    "Cr2O4": "Cr2O3",
                    "Mn2O4": "MnO2",
                }
                spinel_formula = (
                    base_metal
                    + spinel_match.group(3)
                    + "O"
                    + spinel_match.group(4)
                )
                support = spinel_oxide_map.get(spinel_formula, spinel_formula)
                return sm, support, loading

        # Mixed-oxide formula: "Co0.5Ce0.1Al0.4O(sa)", "Fe0.8Ni0.2O"
        mixed_match = re.match(
            r"^([A-Z][a-z]?)\d*\.?\d*(?:[A-Z][a-z]?\d*\.?\d*)+O", name_clean
        )
        if mixed_match:
            first_metal = mixed_match.group(1)
            metal_cat = first_metal if first_metal in KNOWN_METALS else "Other"
            return metal_cat, "mixed-oxide", loading

        return "Other", "Other", loading

    if metal_part is None:
        return "Other", "Other", loading

    # ── Parse metal_part ──
    metal_category = _classify_metal(metal_part)

    # ── Parse support_part ──
    if support_part:
        support_part = re.sub(r"\s*\(.*\)$", "", support_part).strip()
    support = _normalize_support(support_part) if support_part else "Other"

    return metal_category, support, loading


def _looks_like_support(s):
    """Check if a string looks like a bare support (oxide, nitride, carbon)."""
    # Common support prefixes
    for prefix in [
        "CeO2",
        "Al2O3",
        "BN",
        "MgO",
        "SiO2",
        "TiO2",
        "ZrO2",
        "CaO",
        "Y2O3",
        "La2O3",
        "MCM",
        "CNT",
        "SBA",
    ]:
        if s.startswith(prefix):
            return True
    return False


def _classify_metal(metal_str):
    """Classify a metal string into a category."""
    if not metal_str:
        return "Other"

    ms = metal_str.strip()

    # Strip loading prefixes that might remain
    ms = re.sub(r"\d+\.?\d*(?:pct|%)\s*", "", ms).strip()

    # Check for bimetallic with dash: "Fe-Ni", "Co-Mo", "Ru-Ni", "Ru-K"
    if "-" in ms:
        parts = [p.strip() for p in ms.split("-") if p.strip()]
        metals = [p for p in parts if p in KNOWN_METALS]
        non_promoter_metals = [p for p in metals if p not in KNOWN_PROMOTERS]

        if len(non_promoter_metals) == 2:
            return "".join(sorted(non_promoter_metals))
        elif len(non_promoter_metals) == 1:
            return non_promoter_metals[0]
        elif len(metals) >= 1:
            return metals[0]

    # Concatenated bimetallic: "FeCo", "FeNi", "Ni5Co5", "Ni7Co3"
    bimetal_match = re.match(r"([A-Z][a-z]?)\d*([A-Z][a-z]?)\d*$", ms)
    if bimetal_match:
        m1, m2 = bimetal_match.group(1), bimetal_match.group(2)
        if m1 in KNOWN_METALS and m2 in KNOWN_METALS:
            if m1 not in KNOWN_PROMOTERS and m2 not in KNOWN_PROMOTERS:
                return "".join(sorted((m1, m2)))
            elif m1 not in KNOWN_PROMOTERS:
                return m1
            elif m2 not in KNOWN_PROMOTERS:
                return m2

    # "Ru3Fe" pattern
    ru3fe = re.match(r"([A-Z][a-z]?)\d+([A-Z][a-z]?)$", ms)
    if ru3fe:
        m1, m2 = ru3fe.group(1), ru3fe.group(2)
        if m1 in KNOWN_METALS and m2 in KNOWN_METALS:
            return "".join(sorted((m1, m2)))

    # Single metal
    single = re.match(r"([A-Z][a-z]?)\d*$", ms)
    if single and single.group(1) in KNOWN_METALS:
        return single.group(1)

    # Check if starts with a known metal
    for m in sorted(KNOWN_METALS, key=len, reverse=True):
        if ms.startswith(m):
            return m

    return "Other"


def _normalize_support(support_str):
    """Normalize support name to a canonical form."""
    if not support_str:
        return "Other"

    s = support_str.strip()

    # Strip parenthetical notes
    s = re.sub(r"\s*\(.*\)$", "", s)

    # Strip morphology/prefix tags: f-SiO2→SiO2, CeO2-S→CeO2, CeO2-R→CeO2
    # But NOT composite supports like "CeO2-BN", "Y2O3-BN"
    s_base = re.sub(r"^f-", "", s)  # f-SiO2 → SiO2
    s_base = re.sub(
        r"[-_]([SRCHK])$", "", s_base
    )  # -S, -R, -C, -H, -K suffixes
    s_base = re.sub(r"(NR|NP|NC)(-v)?$", "", s_base)  # CeO2NR, CeO2NR-v → CeO2

    # Check explicit perovskite list
    for p in PEROVSKITES:
        if p in s_base:
            return "perovskite"

    # Generic perovskite detection: ABO3
    if _PEROVSKITE_RE.match(s_base):
        return "perovskite"

    # Normalize common support names (check s_base first, then s)
    support_map = {
        "SiO2": "SiO2",
        "CeO2": "CeO2",
        "Al2O3": "Al2O3",
        "MgO": "MgO",
        "CaO": "CaO",
        "MCM-41": "MCM-41",
        "MCM41": "MCM-41",
        "CNTs": "CNTs",
        "CNT": "CNTs",
        "MWCNT": "CNTs",
        "MgAl2O4": "MgAl2O4",
        "TiCSiC": "TiCSiC",
        "BN": "BN",
        "Mo2N": "Mo2N",
        "ZrO2": "ZrO2",
        "SrO": "SrO",
        "TiO2": "TiO2",
        "Y2O3": "Y2O3",
        "La2O3": "La2O3",
        "Nb2O5": "Nb2O5",
        "MnO2": "MnO2",
        "WO3": "WO3",
        "SnO2": "SnO2",
        "SBA-15": "SBA-15",
        "SBA15": "SBA-15",
    }

    for key, val in support_map.items():
        if key in s_base:
            return val

    # Handle composite supports with hyphens: "CeO2-BN", "Y2O3-BN"
    # (checked AFTER support_map so single-component suffixes like -K
    # are already stripped)
    if "-" in s_base:
        # Check if it's a true composite
        # (both parts are known supports/materials)
        parts = s_base.split("-", 1)
        lhs_known = any(k in parts[0] for k in support_map)
        rhs_known = any(k in parts[1] for k in support_map)
        if lhs_known and rhs_known:
            return s_base  # e.g., "CeO2-BN", "Y2O3-BN"

    # Ce-Zr mixed oxides
    if re.match(r"Ce\d*\.?\d*Zr\d*\.?\d*O2", s_base):
        return "CeZrO2"

    # Generic multi-metal oxide: e.g., "Al0.5La0.3Ce0.7"
    # (with or without trailing O)
    if re.match(r"^(?:[A-Z][a-z]?\d*\.?\d*){2,}(?:O\d*)?$", s_base):
        # Check it has at least 2 uppercase letters (i.e., 2+ elements)
        if len(re.findall(r"[A-Z]", s_base)) >= 2:
            return "mixed-oxide"

    # Generic oxide fallback: anything that looks like MetalOx
    if re.match(r"^[A-Z][a-z]?\d*O\d*$", s_base):
        return s_base

    # Spinel support: "Al2O4" → "Al2O3"
    # (from MetalAl2O4 after metal was stripped)
    if re.match(r"^[A-Z][a-z]?\d+O\d+$", s_base):
        return s_base

    return "Other"


# ══════════════════════════════════════════════════════════════════════════
# SECTION 2b — Synthesis Strategy Classification
# ══════════════════════════════════════════════════════════════════════════


def classify_synthesis_strategy(synth_method, actions, reduction_T):  # noqa: N803
    """Classify a material's synthesis route into a strategy category.

    Uses the LLM-extracted synthesis_method field first, falls back to
    action-keyword heuristics if the method is missing or 'other'.
    """
    # ── 1. Try the LLM-extracted synthesis_method first ──
    if synth_method:
        m = synth_method.lower().strip()
        if m != "other":
            if "impregnation" in m or "impregnate" in m:
                return "Impregnation"
            if (
                "coprecipitation" in m
                or "co-precipitation" in m
                or "precipitation" in m
            ):
                return "Co-precipitation"
            if "sol-gel" in m or "sol gel" in m:
                return "Sol-gel"
            if "hydrothermal" in m or "solvothermal" in m:
                return "Hydrothermal"
            if "mechanical" in m or "ball mill" in m or "solid-state" in m:
                return "Solid-state"
            if "combustion" in m:
                return "Combustion"
            # Recognised but not in the main categories — keep as-is
            if m not in ("other", ""):
                return "Other"

    # ── 2. Fallback: classify from action keywords ──
    if not actions:
        return "Other"
    actions_set = set(actions)

    has_precip = "precipitate" in actions_set
    has_impreg = "impregnate" in actions_set
    has_age = "age" in actions_set
    has_reduce = reduction_T is not None and not np.isnan(reduction_T)

    if has_precip and has_age:
        return "Sol-gel"
    if has_precip:
        return "Co-precipitation"
    if has_impreg or actions_set >= {"dissolve", "mix", "dry", "calcine"}:
        return "Impregnation"
    if not has_reduce and "calcine" in actions_set:
        return "Oxide-only"
    if has_reduce:
        return "Impregnation"

    return "Other"


# ══════════════════════════════════════════════════════════════════════════
# SECTION 2c — LLM-Based Material Name Parsing (optional, cached)
# ══════════════════════════════════════════════════════════════════════════

MATERIAL_CACHE_FILE = "material_name_cache.json"

_LLM_PARSE_PROMPT = """\
You are a materials science expert. Parse each material name into its
components.

For EACH material name, return a JSON object with:
- "metal": the active metal or element category (e.g. "Ru", "Ni", "Fe", "Co").
  For bimetallics combine them without spaces (e.g. "FeNi", "CoMo", "RuFe").
  If a component is a known promoter (K, Na, Ca, Ba, Sr, Cs, Li, La, Ce, Nd,
  Sm, Gd, Pr, Y) paired with a catalytic metal, only return the catalytic metal.
  Use "Other" if unknown.
- "support": the support or substrate material in standard chemical formula form
  (e.g. "Al2O3", "CeO2", "MgO", "SiO2", "BN", "CNTs", "MgAl2O4"). For
  composite supports use hyphen (e.g. "CeO2-BN", "Y2O3-BN"). Use "perovskite"
  for perovskite materials. Use "mixed-oxide" for complex multi-metal oxides.
  Use "Other" if unknown or no support.
- "loading": the metal loading as a number (wt%), or null if not specified.

Return ONLY a JSON object mapping each input name to its parsed result.
No explanation, no markdown fences, just the JSON.
"""


def _collect_all_material_names(data_dir, skip_dirs=frozenset()):
    """Scan all paper JSONs and return a set of unique material names."""
    names = set()
    for paper_dir_name in sorted(os.listdir(data_dir)):
        paper_path = data_dir / paper_dir_name
        if not paper_path.is_dir() or paper_dir_name in skip_dirs:
            continue
        for fname in sorted(os.listdir(paper_path)):
            if fname in SKIP_FILES or not fname.endswith(".json"):
                continue
            fpath = paper_path / fname
            try:
                with open(fpath, encoding="utf-8") as f:
                    d = json.load(f)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            names.add(d.get("material", fname.replace(".json", "")))
    return names


def _load_material_cache(data_dir):
    """Load cached LLM-parsed material names from disk."""
    cache_path = data_dir / MATERIAL_CACHE_FILE
    if cache_path.exists():
        with open(cache_path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_material_cache(data_dir, cache):
    """Save LLM-parsed material names to disk."""
    cache_path = data_dir / MATERIAL_CACHE_FILE
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


def _call_llm_for_materials(names, model_name="gemini-2.5-flash"):
    """Call the LLM to parse material names in batches. Returns dict."""
    from dotenv import load_dotenv

    env_path = Path("/home/magled/lematerial-llm-synthesis/.env")
    load_dotenv(env_path, override=True)

    from llm_synthesis.utils.dspy_utils import get_llm_from_name

    lm = get_llm_from_name(
        model_name,
        model_kwargs={"temperature": 0.0, "max_tokens": 16000},
        system_prompt=_LLM_PARSE_PROMPT,
    )

    names_list = sorted(names)
    BATCH_SIZE = 40  # noqa: N806  # ~40 names per batch to stay within token limits
    all_parsed = {}

    for i in range(0, len(names_list), BATCH_SIZE):
        batch = names_list[i : i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        total_batches = (len(names_list) + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"    Batch {batch_num}/{total_batches} ({len(batch)} names)...")

        user_msg = "Parse these catalyst material names:\n\n"
        user_msg += json.dumps({n: "?" for n in batch}, indent=2)

        response = lm(prompt=user_msg)

        # Extract text from response
        if isinstance(response, list):
            text = response[0] if response else ""
        else:
            text = str(response)

        # Strip markdown fences if present
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*\n?", "", text)
            text = re.sub(r"\n?```\s*$", "", text)

        try:
            parsed = json.loads(text)
            all_parsed.update(parsed)
        except json.JSONDecodeError as e:
            print(f"    ⚠ JSON parse error in batch {batch_num}: {e}")
            # Fall back to regex for this batch
            for name in batch:
                metal, support, loading = parse_material_name(name)
                all_parsed[name] = {
                    "metal": metal,
                    "support": support,
                    "loading": loading if not np.isnan(loading) else None,
                }

    return all_parsed


def llm_parse_all_materials(
    data_dir, skip_dirs=frozenset(), model_name="gemini-2.5-flash"
):
    """Parse all material names using an LLM, with filesystem caching.

    - Loads existing cache from data_dir/material_name_cache.json
    - Identifies uncached material names
    - Calls the LLM only for new names
    - Saves updated cache
    - Returns the full cache dict
    """
    print("LLM material name parsing...")

    all_names = _collect_all_material_names(data_dir, skip_dirs)
    cache = _load_material_cache(data_dir)

    uncached = all_names - set(cache.keys())

    if not uncached:
        print(
            f"  All {len(all_names)} material names already cached"
            " — no LLM call needed."
        )
        return cache

    print(
        f"  {len(all_names)} total names, {len(cache)} cached, "
        f"{len(uncached)} new → calling {model_name}..."
    )

    new_parsed = _call_llm_for_materials(uncached, model_name)
    cache.update(new_parsed)
    _save_material_cache(data_dir, cache)

    print(
        f"  ✓ Parsed {len(new_parsed)} new names, cache saved "
        f"({len(cache)} total entries)"
    )

    return cache


# ══════════════════════════════════════════════════════════════════════════
# SECTION 2d — Auto-Detect Promoter Pairs
# ══════════════════════════════════════════════════════════════════════════


def _strip_for_comparison(name):
    """Strip loading prefixes, unicode subscripts, parentheticals."""
    subs = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
    s = name.translate(subs)
    s = re.sub(r"\s*\(.*\)$", "", s).strip()
    s = re.sub(r"^\d+\.?\d*\s*(?:wt\.?%?|wtpct|pct|%)\s*", "", s).strip()
    return s


def _extract_promoter(base_name, prom_name):
    """If prom_name is base_name + a promoter, return promoter element string.
    Otherwise return None."""
    base_s = _strip_for_comparison(base_name)
    prom_s = _strip_for_comparison(prom_name)

    if base_s == prom_s:
        return None

    # Strategy 1: Promoter as prefix slash — "5%La/Ni/Al2O3" vs "Ni/Al2O3"
    prom_parts = prom_s.split("/")
    base_parts = base_s.split("/")
    if len(prom_parts) == 3 and len(base_parts) == 2:
        candidate_base = prom_parts[1] + "/" + prom_parts[2]
        if candidate_base == base_s or candidate_base == "/".join(base_parts):
            elem = re.match(r"(?:\d+\.?\d*%?\s*)?([A-Z][a-z]?)", prom_parts[0])
            if elem and elem.group(1) in KNOWN_PROMOTERS:
                return elem.group(1)

    # Strategy 2: Promoter as dash-element in metal part
    # e.g. "Ru-K/CaO" vs "Ru/CaO"
    if "/" in prom_s and "/" in base_s:
        prom_metal, prom_support = prom_s.rsplit("/", 1)
        base_metal, base_support = base_s.rsplit("/", 1)
        if prom_support == base_support and "-" in prom_metal:
            prom_dash = set(
                re.sub(r"\d+\.?\d*%?\s*", "", p) for p in prom_metal.split("-")
            )
            base_dash = set(
                re.sub(r"\d+\.?\d*%?\s*", "", p) for p in base_metal.split("-")
            )
            extra = prom_dash - base_dash
            if len(extra) == 1:
                elem = extra.pop()
                if elem in KNOWN_PROMOTERS:
                    return elem

    # Strategy 3: Promoter as suffix on support
    # e.g. "Ni5Co5/SiO2-K" vs "Ni5Co5/SiO2"
    if "/" in prom_s and "/" in base_s:
        prom_metal, prom_support = prom_s.rsplit("/", 1)
        base_metal, base_support = base_s.rsplit("/", 1)
        # Strip parentheticals from supports for comparison
        prom_sup_clean = re.sub(r"\s*\(.*\)$", "", prom_support)
        base_sup_clean = re.sub(r"\s*\(.*\)$", "", base_support)
        if prom_metal == base_metal or _strip_for_comparison(
            prom_metal
        ) == _strip_for_comparison(base_metal):
            if (
                prom_sup_clean.startswith(base_sup_clean)
                and "-" in prom_sup_clean
            ):
                suffix = prom_sup_clean[len(base_sup_clean) :].lstrip("-")
                elem_match = re.match(r"([A-Z][a-z]?)", suffix)
                if elem_match and elem_match.group(1) in KNOWN_PROMOTERS:
                    return elem_match.group(1)

    return None


def detect_promoter_pairs(df_curves):
    """Auto-detect base -> promoted catalyst pairs within each paper.

    Returns list of (label, base_conv, prom_conv) tuples sorted by delta.
    """
    # Build lookup: (paper_dir, material_name) -> best non-plasma conv_at_500
    conv_lookup = {}
    plasma_mask = (
        df_curves["is_plasma"]
        if "is_plasma" in df_curves.columns
        else pd.Series(False, index=df_curves.index)
    )
    for _, row in df_curves[~plasma_mask].iterrows():
        key = (row["paper_dir"], row["material_name"])
        val = row["conv_at_500"]
        if pd.notna(val):
            conv_lookup[key] = max(conv_lookup.get(key, 0), val)

    # Group material names by paper
    paper_materials = df_curves.groupby("paper_dir")["material_name"].unique()

    pairs = []
    seen = set()  # avoid A->B and B->A duplicates

    for paper, materials in paper_materials.items():
        mat_list = sorted(set(materials))
        for base_name in mat_list:
            for prom_name in mat_list:
                if base_name == prom_name:
                    continue
                pair_key = (paper, base_name, prom_name)
                if pair_key in seen:
                    continue

                promoter = _extract_promoter(base_name, prom_name)
                if promoter is None:
                    continue

                # Mark both directions as seen
                seen.add(pair_key)
                seen.add((paper, prom_name, base_name))

                base_conv = conv_lookup.get((paper, base_name))
                prom_conv = conv_lookup.get((paper, prom_name))
                if base_conv is None or prom_conv is None:
                    continue

                # Build short label
                base_short = _strip_for_comparison(base_name)
                if "/" in base_short:
                    base_short = (
                        base_short.split("/", 1)[0]
                        + "/"
                        + base_short.split("/")[-1]
                    )
                label = f"{promoter} → {base_short}"
                pairs.append((label, base_conv, prom_conv))

    # Sort by delta (largest promoter effect first)
    pairs.sort(key=lambda x: x[2] - x[1], reverse=True)
    return pairs


# ══════════════════════════════════════════════════════════════════════════
# SECTION 3 — Legend Helpers
# ══════════════════════════════════════════════════════════════════════════


def _sorted_metal_legend(metals_present):
    """Return sorted list: preferred metals first, then alphabetically."""
    preferred_order = [
        "Ru",
        "Ni",
        "Co",
        "Fe",
        "Mo",
        "Pt",
        "Pd",
        "W",
        "Ir",
        "NiCo",
        "FeCo",
        "FeNi",
        "CoNi",
        "RuFe",
        "CoMo",
        "NiRu",
        "RuNi",
    ]
    known = [m for m in preferred_order if m in metals_present]
    extra = sorted(
        m for m in metals_present if m not in preferred_order and m != "Other"
    )
    result = known + extra
    if "Other" in metals_present:
        result.append("Other")
    return result


def _sorted_support_legend(supports_present):
    """Return sorted list: preferred supports first, then alphabetically."""
    preferred_order = [
        "SiO2",
        "CeO2",
        "MgO",
        "Al2O3",
        "CaO",
        "MCM-41",
        "CNTs",
        "MgAl2O4",
        "TiCSiC",
        "BN",
        "Mo2N",
        "perovskite",
        "Y2O3",
        "ZrO2",
        "TiO2",
        "La2O3",
    ]
    known = [s for s in preferred_order if s in supports_present]
    extra = sorted(
        s for s in supports_present if s not in preferred_order and s != "Other"
    )
    result = known + extra
    if "Other" in supports_present:
        result.append("Other")
    return result


# ══════════════════════════════════════════════════════════════════════════
# SECTION 4 — Figure Functions
# ══════════════════════════════════════════════════════════════════════════


# ── Landscape figure sizes: "default" (rectangle) or "square" ──────────
# These are PLOT-AREA sizes (the axes box itself), not overall figure size —
# see _make_axes_fixed_plot_area, which pads the figure so labels/legend sit
# outside this box instead of shrinking it.
LANDSCAPE_FIGSIZE = {
    "default": (10, 6),
    "square": (3, 3),
}

# Fig2b heatmap plot-area sizes — smaller "square" default (3x3in) makes
# per-cell/tick labels read larger for a journal figure.
HEATMAP_FIGSIZE = {
    "default": (10, 6),
    "square": (3, 3),
    "square-mod": (4, 4),
}


def _make_axes_fixed_plot_area(plot_w, plot_h, legend_w=0.0):
    """Create a figure where the axes (plot) area is exactly plot_w x plot_h
    inches, regardless of tick labels / axis labels / legend.

    Port of icicle.utils.visualization.style.make_fig's padding trick: size
    the whole figure larger than the requested plot area, then position the
    axes inside it via subplots_adjust so only the axes box — not labels or
    an outside-right legend — ends up at the requested size. Pass legend_w
    (inches) when a legend sits outside the axes via bbox_to_anchor=(>1, y).
    """
    pad_left = 0.65  # y-axis label + tick labels
    pad_right = 0.15 + legend_w  # right margin (+ outside legend, if any)
    pad_bottom = 0.55  # x-axis label + tick labels
    pad_top = 0.15  # top margin (no title by default)

    fig_w = plot_w + pad_left + pad_right
    fig_h = plot_h + pad_bottom + pad_top

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.subplots_adjust(
        left=pad_left / fig_w,
        right=1.0 - pad_right / fig_w,
        bottom=pad_bottom / fig_h,
        top=1.0 - pad_top / fig_h,
    )
    return fig, ax


def _clean_landscape_df(df_curves, require_support=False):
    """Shared row filter for the conversion-landscape figures (1/6/7)."""
    df = df_curves[
        (~df_curves["is_plasma"])
        & (df_curves["metal"].notna())
        & (df_curves["metal"] != "None")
        & (df_curves["metal"].astype(str) != "nan")
    ].copy()
    if require_support:
        df = df[
            (df["support"].notna())
            & (df["support"].astype(str) != "nan")
            & (df["support"] != "Other")
        ]
    return df


def make_landscape_fig(df_curves, color_by="metal", size="default"):
    """Conversion-landscape line chart, colored by one categorical field.

    One shared 2D template for what used to be three different figures
    (fig1's metal+marker overload, fig6's synthesis-strategy variant, fig7's
    3D waterfall) — same axes/legend/line style, only the color-by field and
    filename change, so the three read as variants of one figure rather than
    three different chart types.

    Args:
        color_by: "metal", "support", or "synthesis".
        size: "default" (10x6) or "square" (8x8) — this is the plot (axes)
            area only; the figure is padded further for labels/legend, so
            the plot area stays this size regardless of legend length.
    """
    color_by_config = {
        "metal": {
            "column": "metal",
            "color_fn": get_metal_color,
            "sort_fn": _sorted_metal_legend,
            "legend_title": "Active Metal",
            "filename": "fig1_conversion_landscape",
        },
        "support": {
            "column": "support",
            "color_fn": get_support_color,
            "sort_fn": _sorted_support_legend,
            "legend_title": "Support",
            "filename": "fig7_conversion_by_support",
        },
        "synthesis": {
            "column": "strategy",
            "color_fn": get_strategy_color,
            "sort_fn": lambda present: [
                s for s in STRATEGY_ORDER if s in present
            ],
            "legend_title": "Synthesis Strategy",
            "filename": "fig6_conversion_by_synthesis",
        },
    }
    if color_by not in color_by_config:
        raise ValueError(f"color_by must be one of {list(color_by_config)}")
    cfg = color_by_config[color_by]

    if df_curves.empty or "is_plasma" not in df_curves.columns:
        print(f"  ⚠ No data for landscape figure (color_by={color_by})")
        return

    df = _clean_landscape_df(df_curves, require_support=(color_by == "support"))
    if df.empty or cfg["column"] not in df.columns:
        print(f"  ⚠ No data for landscape figure (color_by={color_by})")
        return

    plot_w, plot_h = LANDSCAPE_FIGSIZE[size]
    fig, ax = _make_axes_fixed_plot_area(plot_w, plot_h, legend_w=1.6)

    categories_present = set()
    for _, row in df.iterrows():
        coords = np.array(row["coordinates"], dtype=float)
        if len(coords) < 2:
            continue
        temps, convs = coords[:, 0], coords[:, 1]

        cat = row[cfg["column"]]
        if pd.isna(cat):
            cat = "Other"
        color = cfg["color_fn"](cat)

        ax.plot(temps, convs, color=color, alpha=0.55, linewidth=1.0)
        categories_present.add(cat)

    order = cfg["sort_fn"](categories_present)
    handles = [
        Line2D([0], [0], color=cfg["color_fn"](c), lw=2, label=c) for c in order
    ]
    ax.legend(
        handles=handles,
        title=cfg["legend_title"],
        loc="center left",
        frameon=False,
        bbox_to_anchor=(1.02, 0.5),
        fontsize=8,
        title_fontsize=9,
    )

    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel(Y_LABEL)
    ax.set_ylim(-2, 105)

    fig.savefig(OUT_DIR / f"{cfg['filename']}.png")
    fig.savefig(OUT_DIR / f"{cfg['filename']}.pdf")
    print(
        f"  ✓ Landscape (color_by={color_by}) saved"
        f" ({len(df)} curves, {len(order)} {color_by} categories)"
    )
    return fig


def make_fig1(df_curves, size="default"):
    """Figure 1: Cross-paper performance landscape, colored by active metal."""
    return make_landscape_fig(df_curves, color_by="metal", size=size)


def make_metal_zoom_fig(df_curves, metal, size="default"):
    """Zoom into one metal's conversion curves, colored by synthesis strategy.

    Same landscape template as Figure 1/6/7, pre-filtered to a single metal
    (e.g. "Ni") so promoter/support variation within that metal's curves is
    readable — the same idea as filtering Figure 1 down to one row.
    """
    if df_curves.empty or "is_plasma" not in df_curves.columns:
        print(f"  ⚠ No data for metal zoom ({metal})")
        return
    df = _clean_landscape_df(df_curves)
    df = df[df["metal"] == metal]
    if df.empty:
        print(f"  ⚠ No curves for metal={metal}")
        return

    plot_w, plot_h = LANDSCAPE_FIGSIZE[size]
    fig, ax = _make_axes_fixed_plot_area(plot_w, plot_h)

    strategies_present = set()
    for _, row in df.iterrows():
        coords = np.array(row["coordinates"], dtype=float)
        if len(coords) < 2:
            continue
        temps, convs = coords[:, 0], coords[:, 1]
        strat = row.get("strategy", "Other")
        if pd.isna(strat):
            strat = "Other"
        color = get_strategy_color(strat)
        ax.plot(
            temps,
            convs,
            color=color,
            alpha=0.7,
            linewidth=1.2,
            marker="o",
            markersize=3,
        )
        strategies_present.add(strat)

    order = [s for s in STRATEGY_ORDER if s in strategies_present]
    handles = [
        Line2D([0], [0], color=get_strategy_color(s), lw=2, label=s)
        for s in order
    ]
    ax.legend(
        handles=handles,
        title="Synthesis strategy",
        loc="lower right",
        frameon=False,
        fontsize=8,
        title_fontsize=9,
    )

    ax.text(
        0.05,
        0.95,
        f"{metal} (n={len(df)})",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel(Y_LABEL)
    ax.set_ylim(-2, 105)

    fname = f"fig_zoom_{metal}"
    fig.savefig(OUT_DIR / f"{fname}.png")
    fig.savefig(OUT_DIR / f"{fname}.pdf")
    print(f"  ✓ Metal zoom saved (metal={metal}, {len(df)} curves)")
    return fig


def make_fig2(df_curves):
    """Figure 2: Metal x Support heatmap — best conversion at 500°C."""
    if df_curves.empty or "is_plasma" not in df_curves.columns:
        print("  ⚠ No data for Figure 2")
        return
    df = df_curves[
        (~df_curves["is_plasma"])
        & (df_curves["metal"].notna())
        & (df_curves["metal"].astype(str) != "nan")
        & (df_curves["support"].notna())
        & (df_curves["support"].astype(str) != "nan")
        & (df_curves["conv_at_500"].notna())
    ].copy()

    if df.empty:
        print("  ⚠ No data for Figure 2")
        return

    best = df.groupby(["metal", "support"])["conv_at_500"].max().reset_index()

    metals = sorted(
        best["metal"].unique(), key=lambda x: (x not in KNOWN_METALS, x)
    )
    supports = sorted(best["support"].unique())

    data = np.full((len(metals), len(supports)), np.nan)
    for _, row in best.iterrows():
        i = metals.index(row["metal"])
        j = supports.index(row["support"])
        data[i, j] = row["conv_at_500"]

    fig, ax = plt.subplots(
        figsize=(max(8, len(supports) * 0.9), max(4, len(metals) * 0.55))
    )

    from matplotlib.colors import LinearSegmentedColormap

    cmap_heat = LinearSegmentedColormap.from_list(
        "pal_seq", [PAL[5], PAL[2], PAL[12]], N=256
    )
    cmap_heat.set_bad(color=PAL[8])

    im = ax.imshow(data, cmap=cmap_heat, vmin=0, vmax=100, aspect="auto")

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
                    fontsize=8,
                    color="#cccccc",
                )
            else:
                txt_color = "white" if val > 70 else "black"
                ax.text(
                    j,
                    i,
                    f"{val:.0f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    fontweight="bold",
                    color=txt_color,
                )

    ax.set_xticks(range(len(supports)))
    ax.set_xticklabels(supports, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(metals)))
    ax.set_yticklabels(metals, fontsize=9)
    ax.set_xlabel("Support Material")
    ax.set_ylabel("Active Metal / Alloy")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label(f"Best {METRIC_NAME} at {REF_TEMP:.0f} °C (%)")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig2_metal_support_heatmap.png")
    fig.savefig(OUT_DIR / "fig2_metal_support_heatmap.pdf")
    print(
        f"  ✓ Figure 2 saved ({len(metals)} metals × {len(supports)} supports)"
    )
    return fig


def make_fig2b_metal_temp_heatmap(df_curves, temp_bins=None, size="square"):
    """Figure 2b: Metal x Temperature heatmap — median conversion per bin.

    Same "collapsed landscape" idea as Figure 2 (metal x support), but
    collapsing Figure 1's landscape along temperature instead of support:
    one row per metal, one column per temperature bin, cell = median
    conversion of all curves for that metal interpolated at that temperature.
    Rows ordered monometallic-then-bimetallic, alphabetical within each group
    ("Other" last), for journal-article readability.

    Args:
        temp_bins: explicit list of temperatures to bin at, or None to
            auto-derive from the data's range.
        size: "square" (3x3 in plot area, larger relative labels) or
            "default" (10x6) — this is the plot (axes) area only, same as
            make_landscape_fig's size param.
    """
    if df_curves.empty or "is_plasma" not in df_curves.columns:
        print("  ⚠ No data for Figure 2b")
        return
    df = df_curves[
        (~df_curves["is_plasma"])
        & (df_curves["metal"].notna())
        & (df_curves["metal"].astype(str) != "nan")
    ].copy()
    if df.empty:
        print("  ⚠ No data for Figure 2b")
        return

    if temp_bins is None:
        all_temps = np.concatenate(
            [np.array(c, dtype=float)[:, 0] for c in df["coordinates"]]
        )
        lo = 25 * round(all_temps.min() / 25)
        hi = 25 * round(all_temps.max() / 25)
        temp_bins = list(np.arange(lo, hi + 1, 50))

    df["_interp"] = [
        {t: interpolate_at_temp(coords, t) for t in temp_bins}
        for coords in df["coordinates"]
    ]

    def _metal_sort_key(m):
        # (Other last, bimetallic after monometallic, then alphabetical)
        return (m == "Other", m not in KNOWN_METALS, m)

    metals = sorted(df["metal"].unique(), key=_metal_sort_key)
    data = np.full((len(metals), len(temp_bins)), np.nan)
    for i, metal in enumerate(metals):
        sub = df[df["metal"] == metal]
        for j, t in enumerate(temp_bins):
            vals = [d[t] for d in sub["_interp"] if not np.isnan(d[t])]
            if vals:
                data[i, j] = np.median(vals)

    plot_w, plot_h = HEATMAP_FIGSIZE[size]
    fig, ax = _make_axes_fixed_plot_area(plot_w, plot_h, legend_w=0.9)

    from matplotlib.colors import LinearSegmentedColormap

    cmap_heat = LinearSegmentedColormap.from_list(
        "pal_seq", [PAL[5], PAL[2], PAL[12]], N=256
    )
    cmap_heat.set_bad(color=PAL[8])

    im = ax.imshow(data, cmap=cmap_heat, vmin=0, vmax=100, aspect="auto")

    for i in range(len(metals)):
        for j in range(len(temp_bins)):
            val = data[i, j]
            if np.isnan(val):
                continue
            txt_color = "white" if val > 70 else "black"
            ax.text(
                j,
                i,
                f"{val:.0f}",
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold",
                color=txt_color,
            )

    tick_fs, label_fs = (
        (10, 11) if size in ("square", "square-mod") else (8, 10)
    )
    ax.set_xticks(range(len(temp_bins)))
    ax.set_xticklabels([f"{t:.0f}" for t in temp_bins], fontsize=tick_fs)
    ax.set_yticks(range(len(metals)))
    ax.set_yticklabels(metals, fontsize=tick_fs)
    ax.set_xlabel("Temperature (°C)", fontsize=label_fs)
    ax.set_ylabel("Active Metal / Alloy", fontsize=label_fs)

    # Colorbar as its own fixed-width axes (inside the legend_w padding
    # reserved by _make_axes_fixed_plot_area) so it doesn't shrink `ax`.
    fig_w, fig_h = fig.get_size_inches()
    ax_pos = ax.get_position()
    cax = fig.add_axes(
        [ax_pos.x1 + 0.35 / fig_w, ax_pos.y0, 0.15 / fig_w, ax_pos.height]
    )
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label(f"Median {METRIC_NAME} (%)", fontsize=label_fs)
    cbar.ax.tick_params(labelsize=tick_fs)

    fig.savefig(OUT_DIR / "fig2b_metal_temp_heatmap.png")
    fig.savefig(OUT_DIR / "fig2b_metal_temp_heatmap.pdf")
    print(
        f"  ✓ Figure 2b saved ({len(metals)} metals {len(temp_bins)} temp bins)"
    )
    return fig


def make_fig3(df_synthesis):
    """Figure 3: Synthesis action network graph."""
    G = nx.DiGraph()  # noqa: N806
    node_counts = Counter()
    edge_counts = Counter()

    for _, row in df_synthesis.iterrows():
        actions = row["actions"]
        if not actions or len(actions) < 2:
            continue
        for a in actions:
            node_counts[a] += 1
        for a, b in itertools.pairwise(actions):
            edge_counts[(a, b)] += 1

    if not node_counts:
        print("  ⚠ No synthesis data for Figure 3")
        return

    for node, count in node_counts.items():
        G.add_node(node, weight=count)
    for (a, b), count in edge_counts.items():
        G.add_edge(a, b, weight=count)

    fig, ax = plt.subplots(figsize=(12, 8))

    pos = nx.spring_layout(G, seed=42, k=2.5, iterations=100)

    node_sizes = [node_counts[n] * 8 + 200 for n in G.nodes()]
    max_count = max(node_counts.values())
    norm = Normalize(vmin=1, vmax=max_count)
    from matplotlib.colors import LinearSegmentedColormap

    cmap_nodes = LinearSegmentedColormap.from_list(
        "pal_blues", [PAL[3], PAL[2], PAL[12]], N=256
    )

    node_colors = [cmap_nodes(norm(node_counts[n])) for n in G.nodes()]
    edge_widths = [edge_counts[(u, v)] * 0.15 + 0.3 for u, v in G.edges()]

    nx.draw_networkx_edges(
        G,
        pos,
        ax=ax,
        width=edge_widths,
        edge_color="#aaaaaa",
        alpha=0.5,
        arrows=True,
        arrowsize=15,
        connectionstyle="arc3,rad=0.1",
        min_source_margin=15,
        min_target_margin=15,
    )

    nx.draw_networkx_nodes(
        G,
        pos,
        ax=ax,
        node_size=node_sizes,
        node_color=node_colors,
        edgecolors="k",
        linewidths=0.3,
    )

    nx.draw_networkx_labels(G, pos, ax=ax, font_size=7, font_weight="bold")

    min_edge_label = max(
        5,
        sorted(edge_counts.values(), reverse=True)[
            min(10, len(edge_counts) - 1)
        ]
        if len(edge_counts) > 10
        else 1,
    )
    edge_labels = {
        (u, v): str(w)
        for (u, v), w in edge_counts.items()
        if w >= min_edge_label
    }
    if edge_labels:
        nx.draw_networkx_edge_labels(
            G,
            pos,
            edge_labels=edge_labels,
            ax=ax,
            font_size=6,
            font_color="#FF2C00",
            bbox=dict(
                boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.8
            ),
        )

    sm = ScalarMappable(cmap=cmap_nodes, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.5, pad=0.02)
    cbar.set_label("Step frequency across all materials")

    n_papers = df_synthesis["paper_dir"].nunique()
    n_mats = len(df_synthesis)
    ax.set_title(
        f"Synthesis Action Network — {n_mats} materials from {n_papers} papers",
        fontsize=11,
    )
    ax.axis("off")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig3_synthesis_network.png")
    fig.savefig(OUT_DIR / "fig3_synthesis_network.pdf")
    print(
        f"  ✓ Figure 3 saved"
        f" ({len(G.nodes())} actions, {len(G.edges())} transitions)"
    )
    return fig


def make_fig4(df_curves, df_synthesis):
    """Figure 4: Radar charts for top 6 catalysts (by conv at 500°C)."""
    if df_curves.empty or "is_plasma" not in df_curves.columns:
        print("  ⚠ No data for Figure 4")
        return
    df_merged = df_curves.merge(
        df_synthesis[
            [
                "paper_dir",
                "material_name",
                "n_steps",
                "calcination_T",
                "reduction_T",
            ]
        ],
        on=["paper_dir", "material_name"],
        how="inner",
        suffixes=("", "_synth"),
    )

    df_valid = df_merged[
        (df_merged["conv_at_500"].notna())
        & (~df_merged["is_plasma"])
        & (df_merged["n_steps"] >= 2)
        & (df_merged["metal"].notna())
    ].copy()

    if len(df_valid) < 3:
        print("  ⚠ Not enough data for Figure 4")
        return

    top = (
        df_valid.sort_values("conv_at_500", ascending=False)
        .drop_duplicates("material_name")
        .head(6)
    )

    categories = [
        f"{METRIC_NAME.capitalize()}\n@ {REF_TEMP:.0f}°C",
        "Metal\nLoading",
        "Calcination\nTemp.",
        "Reduction\nTemp.",
        "Synthesis\nSteps",
    ]
    N = len(categories)  # noqa: N806

    raw_data = []
    names = []
    for _, row in top.iterrows():
        raw_data.append(
            [
                row["conv_at_500"] if pd.notna(row["conv_at_500"]) else 0,
                row["metal_loading_pct"]
                if pd.notna(row["metal_loading_pct"])
                else 0,
                row["calcination_T"] if pd.notna(row["calcination_T"]) else 0,
                row["reduction_T"] if pd.notna(row["reduction_T"]) else 0,
                row["n_steps"],
            ]
        )
        paper_short = (
            row["paper_dir"].split("_")[-1]
            if "_" in row["paper_dir"]
            else row["paper_dir"]
        )
        names.append(f"{row['material_name']}\n({paper_short})")

    raw = np.array(raw_data, dtype=float)
    mins = raw.min(axis=0)
    maxs = raw.max(axis=0)
    ranges = maxs - mins
    ranges[ranges == 0] = 1

    n_cats = len(top)
    ncols = min(3, n_cats)
    nrows = (n_cats + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.5 * ncols, 4 * nrows),
        subplot_kw=dict(polar=True),
    )
    if n_cats == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    colors = [PAL[2], PAL[3], PAL[4], PAL[0], PAL[12], PAL[13]]

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    for idx in range(len(top)):
        ax = axes[idx]
        norm_vals = [(raw[idx, j] - mins[j]) / ranges[j] for j in range(N)]
        norm_vals += norm_vals[:1]

        c = colors[idx % len(colors)]
        ax.fill(angles, norm_vals, color=c, alpha=0.2)
        ax.plot(angles, norm_vals, color=c, linewidth=1.5)
        ax.scatter(angles[:-1], norm_vals[:-1], color=c, s=25, zorder=5)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=6)
        ax.set_ylim(0, 1.15)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["", "", "", ""], fontsize=5)
        ax.set_title(names[idx], fontsize=7, fontweight="bold", pad=15, color=c)
    for idx in range(len(top), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(
        f"Synthesis–Performance Radar: Top Catalysts"
        f" by {METRIC_NAME} at {REF_TEMP:.0f} °C",
        fontsize=11,
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig4_radar_charts.png")
    fig.savefig(OUT_DIR / "fig4_radar_charts.pdf")
    print(f"  ✓ Figure 4 saved (top {len(top)} catalysts)")
    return fig


def make_fig5a_promoter(df_curves, size="default"):
    """Figure 5a: Promoter effect on performance (auto-detected pairs)."""
    if df_curves.empty or "is_plasma" not in df_curves.columns:
        print("  ⚠ No data for Figure 5a")
        return
    fig, ax1 = plt.subplots(figsize=LANDSCAPE_FIGSIZE[size])

    pair_data = detect_promoter_pairs(df_curves)
    pair_data = pair_data[:12]  # top 12 for readability

    if pair_data:
        pair_data.sort(key=lambda x: x[2] - x[1], reverse=True)
        labels = [p[0] for p in pair_data]
        bases = [p[1] for p in pair_data]
        promoted = [p[2] for p in pair_data]
        deltas = [p - b for b, p in zip(bases, promoted)]

        y_pos = np.arange(len(labels))
        prom_colors = [PAL[i % len(PAL)] for i in range(len(labels))]

        ax1.barh(
            y_pos,
            bases,
            height=0.55,
            color=PAL[8],
            edgecolor="k",
            linewidth=0.3,
            label="Unpromoted",
        )
        ax1.barh(
            y_pos,
            deltas,
            left=bases,
            height=0.55,
            color=prom_colors,
            edgecolor="k",
            linewidth=0.3,
            alpha=0.85,
        )

        for i, (b, p, d) in enumerate(zip(bases, promoted, deltas)):
            sign = "+" if d >= 0 else ""
            ax1.text(
                max(b, p) + 1,
                i,
                f"{sign}{d:.0f}%",
                va="center",
                fontsize=8,
                fontweight="bold",
                color=prom_colors[i],
            )

        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(labels, fontsize=8)
        ax1.set_xlabel(f"{METRIC_NAME.capitalize()} at {REF_TEMP:.0f} °C (%)")
        x_min = max(0, min(bases) - 15)
        ax1.set_xlim(x_min, max(promoted) + 15)
        ax1.legend(
            ["Unpromoted baseline", "Δ from promoter"],
            loc="lower right",
            frameon=False,
            fontsize=7,
        )
    else:
        ax1.text(
            0.5,
            0.5,
            "No promoter pair data found",
            transform=ax1.transAxes,
            ha="center",
            fontsize=10,
            color="grey",
        )

    ax1.set_title(f"Promoter Effect on {METRIC_NAME.capitalize()}", fontsize=10)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig5a_promoter_effect.png")
    fig.savefig(OUT_DIR / "fig5a_promoter_effect.pdf")
    print(
        f"  ✓ Figure 5a saved ({len(pair_data)} promoter pairs auto-detected)"
    )
    return fig


def make_fig5b_conditions(df_curves, df_synthesis, size="default"):
    """Figure 5b: Synthesis conditions (calcination/reduction T) →
    performance scatter."""
    if df_curves.empty or "is_plasma" not in df_curves.columns:
        print("  ⚠ No data for Figure 5b")
        return
    fig, ax2 = plt.subplots(figsize=LANDSCAPE_FIGSIZE[size])

    df_merged = df_curves.merge(
        df_synthesis[
            ["paper_dir", "material_name", "calcination_T", "reduction_T"]
        ],
        on=["paper_dir", "material_name"],
        how="inner",
        suffixes=("", "_synth"),
    )

    df_scatter = df_merged[
        (df_merged["conv_at_500"].notna())
        & (df_merged["calcination_T"].notna())
        & (df_merged["reduction_T"].notna())
        & (~df_merged["is_plasma"])
    ].drop_duplicates("material_name")

    if not df_scatter.empty:
        sizes = df_scatter["metal_loading_pct"].fillna(5) * 8 + 20

        from matplotlib.colors import LinearSegmentedColormap

        cmap_seq = LinearSegmentedColormap.from_list(
            "pal_seq", [PAL[5], PAL[2], PAL[12]], N=256
        )
        scatter = ax2.scatter(
            df_scatter["calcination_T"],
            df_scatter["reduction_T"],
            c=df_scatter["conv_at_500"],
            cmap=cmap_seq,
            s=sizes,
            edgecolors="k",
            linewidths=0.3,
            vmin=0,
            vmax=100,
            alpha=0.8,
            zorder=3,
        )

        cbar = fig.colorbar(scatter, ax=ax2, shrink=0.8, pad=0.02)
        cbar.set_label(f"{METRIC_NAME.capitalize()} at {REF_TEMP:.0f} °C (%)")

        for ml, lab in [(5, "5 wt%"), (20, "20 wt%"), (50, "50 wt%")]:
            ax2.scatter(
                [],
                [],
                s=ml * 8 + 20,
                c="grey",
                alpha=0.5,
                edgecolors="k",
                linewidths=0.3,
                label=lab,
            )
        ax2.legend(
            title="Metal Loading",
            loc="upper left",
            fontsize=7,
            title_fontsize=8,
            frameon=False,
        )
    else:
        ax2.text(
            0.5,
            0.5,
            "No merged synth+perf data",
            transform=ax2.transAxes,
            ha="center",
            fontsize=10,
            color="grey",
        )

    ax2.set_xlabel("Calcination Temperature (°C)")
    ax2.set_ylabel("Reduction Temperature (°C)")
    ax2.set_title("Synthesis Conditions → Performance", fontsize=10)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig5b_synthesis_conditions.png")
    fig.savefig(OUT_DIR / "fig5b_synthesis_conditions.pdf")
    print(f"  ✓ Figure 5b saved ({len(df_scatter)} materials)")
    return fig


def make_fig6(df_curves, size="default"):
    """Figure 6: Conversion landscape, colored by synthesis strategy."""
    return make_landscape_fig(df_curves, color_by="synthesis", size=size)


def make_fig7(df_curves, size="default"):
    """Figure 7: Conversion landscape, colored by support.

    Previously a 3D waterfall (layered by support, colored by metal); now the
    third same-template variant of the landscape chart (metal/synthesis/
    support), colored by support so all three read as one figure family.
    """
    return make_landscape_fig(df_curves, color_by="support", size=size)


def export_landscape_csv(df_curves):
    """Export a companion CSV with one row per curve shown in Fig 1 / Fig 6."""
    df = df_curves[
        (~df_curves["is_plasma"])
        & (df_curves["metal"].notna())
        & (df_curves["metal"] != "None")
        & (df_curves["metal"].astype(str) != "nan")
    ].copy()

    if df.empty:
        print("  ⚠ No data for landscape CSV")
        return

    df["T_min"] = df["coordinates"].apply(lambda c: min(p[0] for p in c))
    df["T_max"] = df["coordinates"].apply(lambda c: max(p[0] for p in c))
    df["n_points"] = df["coordinates"].apply(len)

    out = df[
        [
            "paper_dir",
            "material_name",
            "series_name",
            "metal",
            "support",
            "metal_loading_pct",
            "strategy",
            "conv_at_500",
            "T_min",
            "T_max",
            "n_points",
        ]
    ].rename(columns={"paper_dir": "paper"})

    out = out.sort_values(["paper", "material_name"]).reset_index(drop=True)

    csv_path = OUT_DIR / "landscape_data.csv"
    out.to_csv(csv_path, index=False, float_format="%.1f")
    print(f"  ✓ Landscape CSV saved ({len(out)} rows → {csv_path.name})")


# ══════════════════════════════════════════════════════════════════════════
# SECTION 5 — Debug / Inventory
# ══════════════════════════════════════════════════════════════════════════


def print_debug(df_curves, df_synthesis):
    """Print data inventory for debugging."""
    print("\n" + "=" * 60)
    print("DATA INVENTORY")
    print("=" * 60)

    print(f"\nTotal performance curves: {len(df_curves)}")
    print(f"Total synthesis records:     {len(df_synthesis)}")

    print(
        "\nCurves with conv_at_500:     "
        f"{df_curves['conv_at_500'].notna().sum()}"
    )
    print(
        f"Plasma curves:               {df_curves['is_plasma'].sum() if 'is_plasma' in df_curves.columns else 'N/A'}"  # noqa: E501
    )

    print("\n── Curves per paper ──")
    for paper, count in df_curves["paper_dir"].value_counts().items():
        print(f"  {paper:35s} {count:3d}")

    print("\n── Curves per metal ──")
    for metal, count in df_curves["metal"].value_counts().items():
        print(f"  {metal!s:15s} {count:3d}")

    print("\n── Curves per support ──")
    for sup, count in df_curves["support"].value_counts().head(20).items():
        print(f"  {sup!s:15s} {count:3d}")

    print("\n── Curves per synthesis strategy ──")
    for strat in STRATEGY_ORDER:
        count = (df_curves["strategy"] == strat).sum()
        if count > 0:
            print(f"  {strat:20s} {count:3d}")

    print("\n── Top 10 by conv_at_500 ──")
    top10 = df_curves.nlargest(10, "conv_at_500")
    for _, row in top10.iterrows():
        print(
            f"  {row['conv_at_500']:5.1f}%  {row['material_name']:40s}"
            f"  ({row['paper_dir']})"
        )

    # Parsing failures
    others = df_curves[df_curves["metal"] == "Other"]
    if len(others) > 0:
        print(f"\n── Materials parsed as 'Other' ({len(others)}) ──")
        for _, row in others.iterrows():
            print(f"  {row['material_name']}")

    # Auto-detected promoter pairs
    pairs = detect_promoter_pairs(df_curves)
    if pairs:
        print(f"\n── Auto-detected promoter pairs ({len(pairs)}) ──")
        for label, base_c, prom_c in pairs[:15]:
            delta = prom_c - base_c
            print(
                f"  {label:35s}  {base_c:5.1f}%"
                f" → {prom_c:5.1f}%  (Δ {delta:+.1f}%)"
            )

    print("=" * 60 + "\n")


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate map-of-science figures from LLM-extracted data."
    )
    parser.add_argument(
        "data_dir",
        type=Path,
        help="Path to the results folder containing paper subdirectories",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Print detailed data inventory"
    )
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Use LLM to parse material names (caches results)",
    )
    parser.add_argument(
        "--llm-model",
        default="gemini-2.5-flash",
        help="LLM model for material name parsing (default: gemini-2.5-flash)",
    )
    parser.add_argument(
        "--y-label",
        default=None,
        help="Y-axis label for figures (default: auto-detect or 'Conversion')",
    )
    parser.add_argument(
        "--y-keywords",
        nargs="*",
        default=None,
        help="Keywords to match y_axis_label in plot data (default: ['conversion'])",  # noqa: E501
    )
    parser.add_argument(
        "--ref-temp",
        type=float,
        default=500.0,
        help="Reference temperature for interpolation (default: 500)",
    )
    parser.add_argument(
        "--skip-dirs", nargs="*", default=[], help="Paper directories to skip"
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for figures + CSV (default: <data_dir>/figures)",
    )
    args = parser.parse_args()

    # ── Set global config from CLI ──
    DATA_DIR = args.data_dir.resolve()
    if not DATA_DIR.is_dir():
        parser.error(f"Data directory not found: {DATA_DIR}")

    OUT_DIR = args.out_dir.resolve() if args.out_dir else DATA_DIR / "figures"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.y_keywords:
        Y_KEYWORDS = [kw.lower() for kw in args.y_keywords]
    if args.y_label:
        Y_LABEL = args.y_label
    REF_TEMP = args.ref_temp
    if args.y_label:
        METRIC_NAME = args.y_label.replace("(%)", "").strip()
    skip_dirs = frozenset(args.skip_dirs)

    print(f"Data directory: {DATA_DIR}")
    print(
        f"Metric: {Y_LABEL} | keywords: {Y_KEYWORDS} | ref temp: {REF_TEMP}°C"
    )

    # ── Optionally use LLM for material name parsing ──
    mat_cache = None
    if args.use_llm:
        mat_cache = llm_parse_all_materials(
            data_dir=DATA_DIR,
            skip_dirs=skip_dirs,
            model_name=args.llm_model,
        )

    print("\nLoading data...")
    df_curves, df_synthesis = load_all_data(
        skip_dirs=skip_dirs,
        material_cache=mat_cache,
        DATA_DIR=DATA_DIR,
    )
    print(f"  {len(df_curves)} performance curves loaded")
    print(f"  {len(df_synthesis)} synthesis records loaded")

    if skip_dirs:
        print(f"  Skipped directories: {', '.join(skip_dirs)}")

    if args.debug:
        print_debug(df_curves, df_synthesis)

    print("\nGenerating 9 publication figures...\n")
    for fig in [
        make_fig1(df_curves),
        make_fig2(df_curves),
        make_fig2b_metal_temp_heatmap(df_curves),
        make_fig3(df_synthesis),
        make_fig4(df_curves, df_synthesis),
        make_fig5a_promoter(df_curves),
        make_fig5b_conditions(df_curves, df_synthesis),
        make_fig6(df_curves),
        make_fig7(df_curves),
    ]:
        if fig is not None:
            plt.close(fig)

    print("\nExporting data files...\n")
    export_landscape_csv(df_curves)

    print(f"\nAll outputs saved to {OUT_DIR}")
