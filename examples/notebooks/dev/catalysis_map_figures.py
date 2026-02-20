"""
Map of Science — Publication-Quality Figures from Extracted Data
================================================================
Loads LLM-extracted performance + synthesis data from a results folder
and produces 7 figures (PNG + PDF) + companion CSV.

Fully generic: works with any set of papers and any performance metric.

Run:  uv run python catalysis_map_figures.py /path/to/results_folder
      uv run python catalysis_map_figures.py /path/to/results_folder --debug
      uv run python catalysis_map_figures.py /path/to/results_folder --use-llm
"""

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import networkx as nx
from matplotlib.lines import Line2D
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize


# ── Project style system ─────────────────────────────────────────────────
SRC_DIR = str(Path(__file__).resolve().parents[3] / "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from llm_synthesis.utils.style_utils import set_style, get_palette

set_style("manuscript")

# Override savefig defaults for high-quality output
plt.rcParams.update({"savefig.dpi": 300, "savefig.bbox": "tight"})

# ── Paths (DATA_DIR set from CLI, OUT_DIR defaults to figure_visualisation/) ──
DATA_DIR: Path = None  # set in __main__ from CLI argument
OUT_DIR = Path(__file__).resolve().parents[2] / "data" / "figure_visualisation"

# ── Skip / filter constants ─────────────────────────────────────────────
SKIP_FILES = {
    "linking_summary_human.json", "linking_summary_llm.json",
    "performance_mappings.json", "batch_summary.json", "summary.json",
}

GENERIC_SERIES = {
    "red triangles", "circle markers", "triangle markers", "square markers",
    "green squares", "blue circles", "black diamonds",
    "lower_performing_curve", "middle_performing_curve", "top_performing_curve",
    "catalyst", "plasma+catalyst", "blank", "plasma",
    "plasma on (9095 v)", "thermodynamic equilibrium",
}

# ── Configurable metric settings (set via CLI or auto-detected) ─────────
# These are overridden in __main__ based on --y-label / --ref-temp / auto-detection.
Y_LABEL = "Conversion (%)"          # y-axis label for figures
Y_KEYWORDS = ["conversion"]         # keywords to match y_axis_label in plot_data
REF_TEMP = 500.0                    # reference temperature for interpolation
METRIC_NAME = "conversion"          # short name for filenames and column headers

# ── Known elements for material parsing ─────────────────────────────────
# Active metals commonly found in materials science
KNOWN_METALS = {
    "Ru", "Ni", "Co", "Fe", "Mo", "Pt", "Mn", "Cu",
    "Pd", "W", "Cr", "Re", "Ir", "Rh", "Os", "Au", "Ag",
    "V", "Nb", "Ta", "Ti", "Zr", "Hf",
}

# Elements typically used as promoters (alkali, alkaline earth, rare earth)
KNOWN_PROMOTERS = {
    "K", "Na", "Ca", "Ba", "Sr", "Cs", "Li",
    "La", "Ce", "Nd", "Sm", "Gd", "Pr", "Y",
}

# Known perovskite formulas
PEROVSKITES = {
    "BaTiO3", "SrTiO3", "CaTiO3", "BaZrO3", "SrZrO3", "CaZrO3",
    "BaMnO3", "CaMnO3", "SrMnO3",
    "GdAlO3", "KNbO3", "LaAlO3", "NaNbO3", "SmAlO3",
    "LaFeO3", "LaCoO3", "LaMnO3",
}

# Generic ABO3 perovskite detector
_PEROVSKITE_RE = re.compile(r"^[A-Z][a-z]?[A-Z][a-z]?O3$")

# ── Preferred color / marker assignments ────────────────────────────────
# These are used first; overflow metals/supports get auto-assigned.
_PREFERRED_METAL_COLORS = {
    "Ru":    "#0C5DA5",  # blue
    "Ni":    "#FF9500",  # orange
    "Co":    "#00B945",  # green
    "Fe":    "#FF2C00",  # red
    "Mo":    "#845B97",  # purple
    "NiCo":  "#474747",  # dark grey
    "FeCo":  "#9A607F",  # mauve
    "FeNi":  "#9e9e9e",  # light grey
    "CoNi":  "#F2CC8F",  # sand
    "RuNi":  "#B27EDD",  # light purple
    "RuFe":  "#7B5AEF",  # purple
    "CoMo":  "#E0A2D3",  # pink
    "NiRu":  "#B27EDD",  # light purple
    "Pd":    "#17becf",  # teal
    "Pt":    "#bcbd22",  # olive
    "W":     "#8c564b",  # brown
    "Ir":    "#e377c2",  # pink
    "Other": "#000000",  # black
}

_PREFERRED_SUPPORT_MARKERS = {
    "CeO2":     "s",
    "MgO":      "D",
    "Al2O3":    "^",
    "SiO2":     "o",
    "CaO":      "P",
    "MCM-41":   "v",
    "CNTs":     "*",
    "MgAl2O4":  "h",
    "TiCSiC":   "X",
    "BN":       "p",
    "Mo2N":     "d",
    "perovskite": ">",
    "Y2O3":     "<",
    "ZrO2":     "8",
    "TiO2":     "1",
    "La2O3":    "2",
    "Other":    "o",
}

# Overflow palettes for auto-assignment of new metals/supports
_OVERFLOW_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
    "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5",
]
_OVERFLOW_MARKERS = ["o", "s", "^", "D", "v", "P", "*", "h", "X", "p", "d", ">", "<", "8", "1", "2"]

_auto_metal_colors: dict = {}
_auto_support_markers: dict = {}


def get_metal_color(metal: str) -> str:
    """Return a color for the given metal, auto-assigning if unknown."""
    if metal in _PREFERRED_METAL_COLORS:
        return _PREFERRED_METAL_COLORS[metal]
    if metal not in _auto_metal_colors:
        idx = len(_auto_metal_colors) % len(_OVERFLOW_COLORS)
        _auto_metal_colors[metal] = _OVERFLOW_COLORS[idx]
    return _auto_metal_colors[metal]


def get_support_marker(support: str) -> str:
    """Return a marker for the given support, auto-assigning if unknown."""
    if support in _PREFERRED_SUPPORT_MARKERS:
        return _PREFERRED_SUPPORT_MARKERS[support]
    if support not in _auto_support_markers:
        idx = len(_auto_support_markers) % len(_OVERFLOW_MARKERS)
        _auto_support_markers[support] = _OVERFLOW_MARKERS[idx]
    return _auto_support_markers[support]


# ── Regex patterns ──────────────────────────────────────────────────────
VOLTAGE_RE = re.compile(r"\((\d+)\s*V\)", re.IGNORECASE)  # detect voltage tags in series names
LOADING_RE = re.compile(r"(\d+\.?\d*)\s*(?:wt\.?%|wtpct|pct|%)")

# ── Strategy classification ─────────────────────────────────────────────
STRATEGY_COLORS = {
    "Impregnation":     "#0C5DA5",
    "Co-precipitation": "#00B945",
    "Sol-gel":          "#FF9500",
    "Hydrothermal":     "#845B97",
    "Solid-state":      "#474747",
    "Combustion":       "#FF2C00",
    "Oxide-only":       "#9A607F",
    "Other":            "#9e9e9e",
}
STRATEGY_ORDER = [
    "Impregnation", "Co-precipitation", "Sol-gel", "Hydrothermal",
    "Solid-state", "Combustion", "Oxide-only", "Other",
]


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


def load_all_data(skip_dirs=frozenset(), material_cache=None):
    """Walk all paper directories, load JSONs, return (df_curves, df_synthesis).

    If material_cache is provided (dict mapping material_name → {metal, support, loading}),
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

            synth_rows.append({
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
            })

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

                curves_rows.append({
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
                })

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

    # Strip parenthetical notes BEFORE slash-splitting to avoid "/" inside parens
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
            p0 = re.sub(r"^\d+\.?\d*\s*(?:wt\.?%?|%)\s*", "", slash_parts[0].strip())
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
            return (metal_sym if metal_sym in KNOWN_METALS else "Other"), name_clean, loading

        # Spinel-type: MetalAl2O4, MetalFe2O4, etc.
        spinel_match = re.match(r"^([A-Z][a-z]?)([A-Z][a-z]?)(\d+)O(\d+)$", name_clean)
        if spinel_match:
            sm = spinel_match.group(1)
            base_metal = spinel_match.group(2)
            if sm in KNOWN_METALS:
                # Map spinel to parent oxide: Al2O4→Al2O3, Fe2O4→Fe2O3
                spinel_oxide_map = {"Al2O4": "Al2O3", "Fe2O4": "Fe2O3",
                                    "Cr2O4": "Cr2O3", "Mn2O4": "MnO2"}
                spinel_formula = base_metal + spinel_match.group(3) + "O" + spinel_match.group(4)
                support = spinel_oxide_map.get(spinel_formula, spinel_formula)
                return sm, support, loading

        # Mixed-oxide formula: "Co0.5Ce0.1Al0.4O(sa)", "Fe0.8Ni0.2O"
        mixed_match = re.match(
            r"^([A-Z][a-z]?)\d*\.?\d*(?:[A-Z][a-z]?\d*\.?\d*)+O",
            name_clean
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
    for prefix in ["CeO2", "Al2O3", "BN", "MgO", "SiO2", "TiO2", "ZrO2",
                    "CaO", "Y2O3", "La2O3", "MCM", "CNT", "SBA"]:
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
            return non_promoter_metals[0] + non_promoter_metals[1]
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
                return m1 + m2
            elif m1 not in KNOWN_PROMOTERS:
                return m1
            elif m2 not in KNOWN_PROMOTERS:
                return m2

    # "Ru3Fe" pattern
    ru3fe = re.match(r"([A-Z][a-z]?)\d+([A-Z][a-z]?)$", ms)
    if ru3fe:
        m1, m2 = ru3fe.group(1), ru3fe.group(2)
        if m1 in KNOWN_METALS and m2 in KNOWN_METALS:
            return m1 + m2

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
    s_base = re.sub(r"^f-", "", s)                          # f-SiO2 → SiO2
    s_base = re.sub(r"[-_]([SRCHK])$", "", s_base)          # -S, -R, -C, -H, -K suffixes
    s_base = re.sub(r"(NR|NP|NC)(-v)?$", "", s_base)        # CeO2NR, CeO2NR-v → CeO2

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
        "CeO2": "CeO2", "Al2O3": "Al2O3", "MgO": "MgO",
        "CaO": "CaO", "MCM-41": "MCM-41", "MCM41": "MCM-41",
        "CNTs": "CNTs", "CNT": "CNTs", "MWCNT": "CNTs",
        "MgAl2O4": "MgAl2O4", "TiCSiC": "TiCSiC",
        "BN": "BN", "Mo2N": "Mo2N",
        "ZrO2": "ZrO2", "SrO": "SrO", "TiO2": "TiO2",
        "Y2O3": "Y2O3", "La2O3": "La2O3",
        "Nb2O5": "Nb2O5", "MnO2": "MnO2", "WO3": "WO3", "SnO2": "SnO2",
        "SBA-15": "SBA-15", "SBA15": "SBA-15",
    }

    for key, val in support_map.items():
        if key in s_base:
            return val

    # Handle composite supports with hyphens: "CeO2-BN", "Y2O3-BN"
    # (checked AFTER support_map so single-component suffixes like -K are already stripped)
    if "-" in s_base:
        # Check if it's a true composite (both parts are known supports/materials)
        parts = s_base.split("-", 1)
        lhs_known = any(k in parts[0] for k in support_map)
        rhs_known = any(k in parts[1] for k in support_map)
        if lhs_known and rhs_known:
            return s_base  # e.g., "CeO2-BN", "Y2O3-BN"

    # Ce-Zr mixed oxides
    if re.match(r"Ce\d*\.?\d*Zr\d*\.?\d*O2", s_base):
        return "CeZrO2"

    # Generic multi-metal oxide: e.g., "Al0.5La0.3Ce0.7" (with or without trailing O)
    if re.match(r"^(?:[A-Z][a-z]?\d*\.?\d*){2,}(?:O\d*)?$", s_base):
        # Check it has at least 2 uppercase letters (i.e., 2+ elements)
        if len(re.findall(r"[A-Z]", s_base)) >= 2:
            return "mixed-oxide"

    # Generic oxide fallback: anything that looks like MetalOx
    if re.match(r"^[A-Z][a-z]?\d*O\d*$", s_base):
        return s_base

    # Spinel support: "Al2O4" → "Al2O3" (from MetalAl2O4 after metal was stripped)
    if re.match(r"^[A-Z][a-z]?\d+O\d+$", s_base):
        return s_base

    return "Other"


# ══════════════════════════════════════════════════════════════════════════
# SECTION 2b — Synthesis Strategy Classification
# ══════════════════════════════════════════════════════════════════════════

def classify_synthesis_strategy(synth_method, actions, reduction_T):
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
            if "coprecipitation" in m or "co-precipitation" in m or "precipitation" in m:
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
You are a materials science expert. Parse each material name into its components.

For EACH material name, return a JSON object with:
- "metal": the active metal or element category (e.g. "Ru", "Ni", "Fe", "Co"). For
  bimetallics combine them without spaces (e.g. "FeNi", "CoMo", "RuFe"). If a
  component is a known promoter (K, Na, Ca, Ba, Sr, Cs, Li, La, Ce, Nd, Sm, Gd, Pr,
  Y) paired with a catalytic metal, only return the catalytic metal. Use "Other" if
  unknown.
- "support": the support or substrate material in standard chemical formula form
  (e.g. "Al2O3", "CeO2", "MgO", "SiO2", "BN", "CNTs", "MgAl2O4"). For composite
  supports use hyphen (e.g. "CeO2-BN", "Y2O3-BN"). Use "perovskite" for perovskite
  materials. Use "mixed-oxide" for complex multi-metal oxides. Use "Other" if unknown
  or no support.
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

    env_path = Path(__file__).resolve().parents[3] / ".env"
    load_dotenv(env_path, override=True)

    from llm_synthesis.utils.dspy_utils import get_llm_from_name

    lm = get_llm_from_name(
        model_name,
        model_kwargs={"temperature": 0.0, "max_tokens": 16000},
        system_prompt=_LLM_PARSE_PROMPT,
    )

    names_list = sorted(names)
    BATCH_SIZE = 40  # ~40 names per batch to stay within token limits
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
                    "metal": metal, "support": support,
                    "loading": loading if not np.isnan(loading) else None,
                }

    return all_parsed


def llm_parse_all_materials(data_dir, skip_dirs=frozenset(),
                            model_name="gemini-2.5-flash"):
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
        print(f"  All {len(all_names)} material names already cached — no LLM call needed.")
        return cache

    print(f"  {len(all_names)} total names, {len(cache)} cached, "
          f"{len(uncached)} new → calling {model_name}...")

    new_parsed = _call_llm_for_materials(uncached, model_name)
    cache.update(new_parsed)
    _save_material_cache(data_dir, cache)

    print(f"  ✓ Parsed {len(new_parsed)} new names, cache saved "
          f"({len(cache)} total entries)")

    return cache


# ══════════════════════════════════════════════════════════════════════════
# SECTION 2d — Auto-Detect Promoter Pairs
# ══════════════════════════════════════════════════════════════════════════

def _strip_for_comparison(name):
    """Strip loading prefixes, unicode subscripts, parentheticals for comparison."""
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

    # Strategy 2: Promoter as dash-element in metal part — "Ru-K/CaO" vs "Ru/CaO"
    if "/" in prom_s and "/" in base_s:
        prom_metal, prom_support = prom_s.rsplit("/", 1)
        base_metal, base_support = base_s.rsplit("/", 1)
        if prom_support == base_support and "-" in prom_metal:
            prom_dash = set(re.sub(r"\d+\.?\d*%?\s*", "", p) for p in prom_metal.split("-"))
            base_dash = set(re.sub(r"\d+\.?\d*%?\s*", "", p) for p in base_metal.split("-"))
            extra = prom_dash - base_dash
            if len(extra) == 1:
                elem = extra.pop()
                if elem in KNOWN_PROMOTERS:
                    return elem

    # Strategy 3: Promoter as suffix on support — "Ni5Co5/SiO2-K" vs "Ni5Co5/SiO2"
    if "/" in prom_s and "/" in base_s:
        prom_metal, prom_support = prom_s.rsplit("/", 1)
        base_metal, base_support = base_s.rsplit("/", 1)
        # Strip parentheticals from supports for comparison
        prom_sup_clean = re.sub(r"\s*\(.*\)$", "", prom_support)
        base_sup_clean = re.sub(r"\s*\(.*\)$", "", base_support)
        if prom_metal == base_metal or _strip_for_comparison(prom_metal) == _strip_for_comparison(base_metal):
            if prom_sup_clean.startswith(base_sup_clean) and "-" in prom_sup_clean:
                suffix = prom_sup_clean[len(base_sup_clean):].lstrip("-")
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
    for _, row in df_curves[~df_curves["is_plasma"]].iterrows():
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
                    base_short = base_short.split("/", 1)[0] + "/" + base_short.split("/")[-1]
                label = f"{promoter} → {base_short}"
                pairs.append((label, base_conv, prom_conv))

    # Sort by delta (largest promoter effect first)
    pairs.sort(key=lambda x: x[2] - x[1], reverse=True)
    return pairs


# ══════════════════════════════════════════════════════════════════════════
# SECTION 3 — Legend Helpers
# ══════════════════════════════════════════════════════════════════════════

def _sorted_metal_legend(metals_present):
    """Return sorted list: preferred metals first, then new ones alphabetically."""
    preferred_order = ["Ru", "Ni", "Co", "Fe", "Mo", "Pt", "Pd", "W", "Ir",
                       "NiCo", "FeCo", "FeNi", "CoNi", "RuFe", "CoMo", "NiRu", "RuNi"]
    known = [m for m in preferred_order if m in metals_present]
    extra = sorted(m for m in metals_present if m not in preferred_order and m != "Other")
    result = known + extra
    if "Other" in metals_present:
        result.append("Other")
    return result


def _sorted_support_legend(supports_present):
    """Return sorted list: preferred supports first, then new ones alphabetically."""
    preferred_order = ["SiO2", "CeO2", "MgO", "Al2O3", "CaO", "MCM-41",
                       "CNTs", "MgAl2O4", "TiCSiC", "BN", "Mo2N",
                       "perovskite", "Y2O3", "ZrO2", "TiO2", "La2O3"]
    known = [s for s in preferred_order if s in supports_present]
    extra = sorted(s for s in supports_present if s not in preferred_order and s != "Other")
    result = known + extra
    if "Other" in supports_present:
        result.append("Other")
    return result


# ══════════════════════════════════════════════════════════════════════════
# SECTION 4 — Figure Functions
# ══════════════════════════════════════════════════════════════════════════

def make_fig1(df_curves):
    """Figure 1: Cross-paper performance landscape."""
    df = df_curves[
        (~df_curves["is_plasma"]) &
        (df_curves["metal"].notna()) &
        (df_curves["metal"] != "None") &
        (df_curves["metal"].astype(str) != "nan")
    ].copy()

    if df.empty:
        print("  ⚠ No data for Figure 1")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    metals_present = set()
    supports_present = set()

    for _, row in df.iterrows():
        coords = np.array(row["coordinates"], dtype=float)
        if len(coords) < 2:
            continue
        temps, convs = coords[:, 0], coords[:, 1]

        metal = row["metal"]
        support = row["support"] if pd.notna(row["support"]) else "Other"

        color = get_metal_color(metal)
        marker = get_support_marker(support)

        ax.plot(temps, convs, color=color, alpha=0.55, linewidth=1.0)

        step = max(1, len(temps) // 5)
        ax.scatter(temps[::step], convs[::step], color=color, marker=marker,
                   s=20, zorder=3, edgecolors="white", linewidth=0.3, alpha=0.8)

        metals_present.add(metal)
        supports_present.add(support)

    # Metal legend (inside plot, lower-right)
    metal_order = _sorted_metal_legend(metals_present)
    metal_handles = [Line2D([0], [0], color=get_metal_color(m), lw=2, label=m)
                     for m in metal_order]
    leg1 = ax.legend(handles=metal_handles, title="Active Metal",
                     loc="lower right", frameon=True, framealpha=0.95,
                     edgecolor="grey", fontsize=7, title_fontsize=8)
    ax.add_artist(leg1)

    # Support legend (outside plot on the right)
    sup_order = _sorted_support_legend(supports_present)
    sup_handles = [Line2D([0], [0], marker=get_support_marker(s), color="grey",
                          lw=0, markersize=5, label=s)
                   for s in sup_order]
    ax.legend(handles=sup_handles, title="Support",
              loc="center left", frameon=True, framealpha=0.95,
              edgecolor="grey", bbox_to_anchor=(1.02, 0.5),
              fontsize=7, title_fontsize=8)

    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel(Y_LABEL)
    ax.set_ylim(-2, 105)

    fig.subplots_adjust(right=0.82)
    fig.savefig(OUT_DIR / "fig1_conversion_landscape.png")
    fig.savefig(OUT_DIR / "fig1_conversion_landscape.pdf")
    plt.close(fig)
    print(f"  ✓ Figure 1 saved ({len(df)} curves)")


def make_fig2(df_curves):
    """Figure 2: Metal x Support heatmap — best conversion at 500°C."""
    df = df_curves[
        (~df_curves["is_plasma"]) &
        (df_curves["metal"].notna()) &
        (df_curves["metal"].astype(str) != "nan") &
        (df_curves["support"].notna()) &
        (df_curves["support"].astype(str) != "nan") &
        (df_curves["conv_at_500"].notna())
    ].copy()

    if df.empty:
        print("  ⚠ No data for Figure 2")
        return

    best = df.groupby(["metal", "support"])["conv_at_500"].max().reset_index()

    metals = sorted(best["metal"].unique(),
                    key=lambda x: (x not in KNOWN_METALS, x))
    supports = sorted(best["support"].unique())

    data = np.full((len(metals), len(supports)), np.nan)
    for _, row in best.iterrows():
        i = metals.index(row["metal"])
        j = supports.index(row["support"])
        data[i, j] = row["conv_at_500"]

    fig, ax = plt.subplots(figsize=(max(8, len(supports) * 0.9), max(4, len(metals) * 0.55)))

    cmap_heat = plt.cm.YlOrRd.copy()
    cmap_heat.set_bad(color="#f5f5f5")

    im = ax.imshow(data, cmap=cmap_heat, vmin=0, vmax=100, aspect="auto")

    for i in range(len(metals)):
        for j in range(len(supports)):
            val = data[i, j]
            if np.isnan(val):
                ax.text(j, i, "—", ha="center", va="center",
                        fontsize=8, color="#cccccc")
            else:
                txt_color = "white" if val > 70 else "black"
                ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                        fontsize=8, fontweight="bold", color=txt_color)

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
    plt.close(fig)
    print(f"  ✓ Figure 2 saved ({len(metals)} metals × {len(supports)} supports)")


def make_fig3(df_synthesis):
    """Figure 3: Synthesis action network graph."""
    G = nx.DiGraph()
    node_counts = Counter()
    edge_counts = Counter()

    for _, row in df_synthesis.iterrows():
        actions = row["actions"]
        if not actions or len(actions) < 2:
            continue
        for a in actions:
            node_counts[a] += 1
        for a, b in zip(actions[:-1], actions[1:]):
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
    cmap_nodes = plt.cm.Blues

    node_colors = [cmap_nodes(norm(node_counts[n])) for n in G.nodes()]
    edge_widths = [edge_counts[(u, v)] * 0.15 + 0.3 for u, v in G.edges()]

    nx.draw_networkx_edges(G, pos, ax=ax, width=edge_widths,
                           edge_color="#aaaaaa", alpha=0.5,
                           arrows=True, arrowsize=15,
                           connectionstyle="arc3,rad=0.1",
                           min_source_margin=15, min_target_margin=15)

    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_sizes,
                           node_color=node_colors, edgecolors="white",
                           linewidths=1.2)

    nx.draw_networkx_labels(G, pos, ax=ax, font_size=7, font_weight="bold")

    min_edge_label = max(5, sorted(edge_counts.values(), reverse=True)[min(10, len(edge_counts)-1)]
                         if len(edge_counts) > 10 else 1)
    edge_labels = {(u, v): str(w) for (u, v), w in edge_counts.items()
                   if w >= min_edge_label}
    if edge_labels:
        nx.draw_networkx_edge_labels(
            G, pos, edge_labels=edge_labels, ax=ax,
            font_size=6, font_color="#FF2C00",
            bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.8))

    sm = ScalarMappable(cmap=cmap_nodes, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.5, pad=0.02)
    cbar.set_label("Step frequency across all materials")

    n_papers = df_synthesis["paper_dir"].nunique()
    n_mats = len(df_synthesis)
    ax.set_title(f"Synthesis Action Network — {n_mats} materials from {n_papers} papers",
                  fontsize=11)
    ax.axis("off")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig3_synthesis_network.png")
    fig.savefig(OUT_DIR / "fig3_synthesis_network.pdf")
    plt.close(fig)
    print(f"  ✓ Figure 3 saved ({len(G.nodes())} actions, {len(G.edges())} transitions)")


def make_fig4(df_curves, df_synthesis):
    """Figure 4: Radar charts for top 6 catalysts (by conv at 500°C)."""
    df_merged = df_curves.merge(
        df_synthesis[["paper_dir", "material_name", "n_steps",
                      "calcination_T", "reduction_T"]],
        on=["paper_dir", "material_name"],
        how="inner",
        suffixes=("", "_synth"),
    )

    df_valid = df_merged[
        (df_merged["conv_at_500"].notna()) &
        (~df_merged["is_plasma"]) &
        (df_merged["n_steps"] >= 2) &
        (df_merged["metal"].notna())
    ].copy()

    if len(df_valid) < 3:
        print("  ⚠ Not enough data for Figure 4")
        return

    top = (df_valid.sort_values("conv_at_500", ascending=False)
           .drop_duplicates("material_name")
           .head(6))

    categories = [
        f"{METRIC_NAME.capitalize()}\n@ {REF_TEMP:.0f}°C",
        "Metal\nLoading",
        "Calcination\nTemp.",
        "Reduction\nTemp.",
        "Synthesis\nSteps",
    ]
    N = len(categories)

    raw_data = []
    names = []
    for _, row in top.iterrows():
        raw_data.append([
            row["conv_at_500"] if pd.notna(row["conv_at_500"]) else 0,
            row["metal_loading_pct"] if pd.notna(row["metal_loading_pct"]) else 0,
            row["calcination_T"] if pd.notna(row["calcination_T"]) else 0,
            row["reduction_T"] if pd.notna(row["reduction_T"]) else 0,
            row["n_steps"],
        ])
        paper_short = row["paper_dir"].split("_")[-1] if "_" in row["paper_dir"] else row["paper_dir"]
        names.append(f"{row['material_name']}\n({paper_short})")

    raw = np.array(raw_data, dtype=float)
    mins = raw.min(axis=0)
    maxs = raw.max(axis=0)
    ranges = maxs - mins
    ranges[ranges == 0] = 1

    n_cats = len(top)
    ncols = min(3, n_cats)
    nrows = (n_cats + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4 * nrows),
                              subplot_kw=dict(polar=True))
    if n_cats == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    colors = ["#0C5DA5", "#00B945", "#FF9500", "#FF2C00", "#845B97", "#9A607F"]

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
        ax.grid(alpha=0.3)

    for idx in range(len(top), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(f"Synthesis–Performance Radar: Top Catalysts by {METRIC_NAME} at {REF_TEMP:.0f} °C",
                  fontsize=11, y=1.01)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig4_radar_charts.png")
    fig.savefig(OUT_DIR / "fig4_radar_charts.pdf")
    plt.close(fig)
    print(f"  ✓ Figure 4 saved (top {len(top)} catalysts)")


def make_fig5(df_curves, df_synthesis):
    """Figure 5: Two panels — (a) promoter effect (auto-detected), (b) conditions scatter."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5),
                                     gridspec_kw={"width_ratios": [1, 1.2]})

    # ── Panel A: Auto-detected promoter effect ──
    pair_data = detect_promoter_pairs(df_curves)

    # Limit to top 12 pairs for readability
    pair_data = pair_data[:12]

    if pair_data:
        pair_data.sort(key=lambda x: x[2] - x[1], reverse=True)
        labels = [p[0] for p in pair_data]
        bases = [p[1] for p in pair_data]
        promoted = [p[2] for p in pair_data]
        deltas = [p - b for b, p in zip(bases, promoted)]

        y_pos = np.arange(len(labels))

        prom_cycle = ["#0C5DA5", "#00B945", "#FF9500", "#FF2C00",
                      "#845B97", "#474747", "#9e9e9e", "#9A607F"]
        prom_colors = [prom_cycle[i % len(prom_cycle)] for i in range(len(labels))]

        ax1.barh(y_pos, bases, height=0.55, color="#e0e0e0",
                 edgecolor="white", label="Unpromoted")
        ax1.barh(y_pos, deltas, left=bases, height=0.55,
                 color=prom_colors, edgecolor="white", alpha=0.85)

        for i, (b, p, d) in enumerate(zip(bases, promoted, deltas)):
            sign = "+" if d >= 0 else ""
            ax1.text(max(b, p) + 1, i, f"{sign}{d:.0f}%", va="center",
                     fontsize=8, fontweight="bold", color=prom_colors[i])

        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(labels, fontsize=8)
        ax1.set_xlabel(f"{METRIC_NAME.capitalize()} at {REF_TEMP:.0f} °C (%)")
        x_min = max(0, min(bases) - 15)
        ax1.set_xlim(x_min, max(promoted) + 15)
        ax1.legend(["Unpromoted baseline", "Δ from promoter"],
                   loc="lower right", fontsize=7)
    else:
        ax1.text(0.5, 0.5, "No promoter pair data found",
                 transform=ax1.transAxes, ha="center", fontsize=10, color="grey")

    ax1.set_title(f"(a) Promoter Effect on {METRIC_NAME.capitalize()}", fontsize=10)
    ax1.grid(axis="x", alpha=0.15)

    # ── Panel B: Synthesis conditions → performance scatter ──
    df_merged = df_curves.merge(
        df_synthesis[["paper_dir", "material_name", "calcination_T", "reduction_T"]],
        on=["paper_dir", "material_name"], how="inner",
        suffixes=("", "_synth"),
    )

    df_scatter = df_merged[
        (df_merged["conv_at_500"].notna()) &
        (df_merged["calcination_T"].notna()) &
        (df_merged["reduction_T"].notna()) &
        (~df_merged["is_plasma"])
    ].drop_duplicates("material_name")

    if not df_scatter.empty:
        sizes = df_scatter["metal_loading_pct"].fillna(5) * 8 + 20

        scatter = ax2.scatter(
            df_scatter["calcination_T"], df_scatter["reduction_T"],
            c=df_scatter["conv_at_500"], cmap="RdYlGn",
            s=sizes, edgecolors="white", linewidth=0.5,
            vmin=0, vmax=100, alpha=0.8, zorder=3)

        cbar = fig.colorbar(scatter, ax=ax2, shrink=0.8, pad=0.02)
        cbar.set_label(f"{METRIC_NAME.capitalize()} at {REF_TEMP:.0f} °C (%)")

        for ml, lab in [(5, "5 wt%"), (20, "20 wt%"), (50, "50 wt%")]:
            ax2.scatter([], [], s=ml * 8 + 20, c="grey", alpha=0.5,
                        edgecolors="white", label=lab)
        ax2.legend(title="Metal Loading", loc="upper left", fontsize=7,
                   title_fontsize=8, framealpha=0.9)
    else:
        ax2.text(0.5, 0.5, "No merged synth+perf data",
                 transform=ax2.transAxes, ha="center", fontsize=10, color="grey")

    ax2.set_xlabel("Calcination Temperature (°C)")
    ax2.set_ylabel("Reduction Temperature (°C)")
    ax2.set_title("(b) Synthesis Conditions → Performance", fontsize=10)
    ax2.grid(alpha=0.15)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig5_promoter_and_conditions.png")
    fig.savefig(OUT_DIR / "fig5_promoter_and_conditions.pdf")
    plt.close(fig)
    n_pairs = len(pair_data)
    print(f"  ✓ Figure 5 saved ({n_pairs} promoter pairs auto-detected)")


def make_fig6(df_curves):
    """Figure 6: Conversion landscape colored by synthesis strategy."""
    df = df_curves[
        (~df_curves["is_plasma"]) &
        (df_curves["metal"].notna()) &
        (df_curves["metal"] != "None") &
        (df_curves["metal"].astype(str) != "nan")
    ].copy()

    if df.empty:
        print("  ⚠ No data for Figure 6")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    strategies_present = set()

    for _, row in df.iterrows():
        coords = np.array(row["coordinates"], dtype=float)
        if len(coords) < 2:
            continue
        temps, convs = coords[:, 0], coords[:, 1]

        strat = row.get("strategy", "Other")
        if pd.isna(strat):
            strat = "Other"
        color = STRATEGY_COLORS.get(strat, "#9e9e9e")

        ax.plot(temps, convs, color=color, alpha=0.5, linewidth=1.2)
        strategies_present.add(strat)

    legend_order = [s for s in STRATEGY_ORDER if s in strategies_present]
    handles = [Line2D([0], [0], color=STRATEGY_COLORS[s], lw=2.5, label=s)
               for s in legend_order]
    ax.legend(handles=handles, title="Synthesis Strategy",
              loc="lower right", frameon=True, framealpha=0.95,
              edgecolor="grey", fontsize=8, title_fontsize=9)

    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel(Y_LABEL)
    ax.set_ylim(-2, 105)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig6_conversion_by_synthesis.png")
    fig.savefig(OUT_DIR / "fig6_conversion_by_synthesis.pdf")
    plt.close(fig)

    strat_counts = df["strategy"].value_counts()
    detail = ", ".join(f"{s}: {strat_counts.get(s, 0)}" for s in legend_order)
    print(f"  ✓ Figure 6 saved ({len(df)} curves — {detail})")


def make_fig7(df_curves):
    """Figure 7: 3D waterfall — conversion curves layered by support, colored by metal."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    df = df_curves[
        (~df_curves["is_plasma"]) &
        (df_curves["metal"].notna()) &
        (df_curves["metal"] != "None") &
        (df_curves["metal"].astype(str) != "nan") &
        (df_curves["support"].notna()) &
        (df_curves["support"].astype(str) != "nan") &
        (df_curves["support"] != "Other")
    ].copy()

    if df.empty:
        print("  ⚠ No data for Figure 7")
        return

    median_conv = (
        df[df["conv_at_500"].notna()]
        .groupby("support")["conv_at_500"]
        .median()
        .sort_values(ascending=False)
    )
    support_order = list(median_conv.index)

    support_counts = df["support"].value_counts()
    support_order = [s for s in support_order if support_counts.get(s, 0) >= 2]

    if not support_order:
        print("  ⚠ Not enough support groups for Figure 7")
        return

    support_to_y = {s: i for i, s in enumerate(support_order)}

    fig = plt.figure(figsize=(14, 9))
    ax = fig.add_subplot(111, projection="3d")

    metals_present = set()

    for _, row in df.iterrows():
        support = row["support"]
        if support not in support_to_y:
            continue

        coords = np.array(row["coordinates"], dtype=float)
        if len(coords) < 2:
            continue
        temps, convs = coords[:, 0], coords[:, 1]

        metal = row["metal"]
        color = get_metal_color(metal)
        y_val = support_to_y[support]

        ys = np.full_like(temps, y_val)
        ax.plot(temps, ys, convs, color=color, alpha=0.6, linewidth=0.9)

        metals_present.add(metal)

    ax.set_xlabel("Temperature (°C)", labelpad=10)
    ax.set_zlabel(Y_LABEL, labelpad=8)
    ax.set_ylabel("")

    ax.set_yticks(range(len(support_order)))
    ax.set_yticklabels(support_order, fontsize=7, ha="left")
    ax.set_zlim(-2, 105)

    ax.view_init(elev=25, azim=-55)

    metal_order = _sorted_metal_legend(metals_present)
    handles = [Line2D([0], [0], color=get_metal_color(m), lw=2.5, label=m)
               for m in metal_order]
    ax.legend(handles=handles, title="Active Metal",
              loc="upper left", frameon=True, framealpha=0.95,
              edgecolor="grey", fontsize=7, title_fontsize=8)

    fig.subplots_adjust(left=0.02, right=0.95, bottom=0.05, top=0.98)
    fig.savefig(OUT_DIR / "fig7_3d_waterfall.png", dpi=300)
    fig.savefig(OUT_DIR / "fig7_3d_waterfall.pdf")
    plt.close(fig)

    n_supports = len(support_order)
    n_curves = sum(1 for _, r in df.iterrows() if r["support"] in support_to_y)
    print(f"  ✓ Figure 7 saved ({n_curves} curves across {n_supports} supports)")


def export_landscape_csv(df_curves):
    """Export a companion CSV with one row per curve shown in Fig 1 / Fig 6."""
    df = df_curves[
        (~df_curves["is_plasma"]) &
        (df_curves["metal"].notna()) &
        (df_curves["metal"] != "None") &
        (df_curves["metal"].astype(str) != "nan")
    ].copy()

    if df.empty:
        print("  ⚠ No data for landscape CSV")
        return

    df["T_min"] = df["coordinates"].apply(lambda c: min(p[0] for p in c))
    df["T_max"] = df["coordinates"].apply(lambda c: max(p[0] for p in c))
    df["n_points"] = df["coordinates"].apply(len)

    out = df[[
        "paper_dir", "material_name", "series_name",
        "metal", "support", "metal_loading_pct",
        "strategy", "conv_at_500",
        "T_min", "T_max", "n_points",
    ]].rename(columns={"paper_dir": "paper"})

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

    print(f"\nCurves with conv_at_500:     {df_curves['conv_at_500'].notna().sum()}")
    print(f"Plasma curves:               {df_curves['is_plasma'].sum()}")

    print("\n── Curves per paper ──")
    for paper, count in df_curves["paper_dir"].value_counts().items():
        print(f"  {paper:35s} {count:3d}")

    print("\n── Curves per metal ──")
    for metal, count in df_curves["metal"].value_counts().items():
        print(f"  {str(metal):15s} {count:3d}")

    print("\n── Curves per support ──")
    for sup, count in df_curves["support"].value_counts().head(20).items():
        print(f"  {str(sup):15s} {count:3d}")

    print("\n── Curves per synthesis strategy ──")
    for strat in STRATEGY_ORDER:
        count = (df_curves["strategy"] == strat).sum()
        if count > 0:
            print(f"  {strat:20s} {count:3d}")

    print("\n── Top 10 by conv_at_500 ──")
    top10 = df_curves.nlargest(10, "conv_at_500")
    for _, row in top10.iterrows():
        print(f"  {row['conv_at_500']:5.1f}%  {row['material_name']:40s}  ({row['paper_dir']})")

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
            print(f"  {label:35s}  {base_c:5.1f}% → {prom_c:5.1f}%  (Δ {delta:+.1f}%)")

    print("=" * 60 + "\n")


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate map-of-science figures from LLM-extracted data."
    )
    parser.add_argument("data_dir", type=Path,
                        help="Path to the results folder containing paper subdirectories")
    parser.add_argument("--debug", action="store_true",
                        help="Print detailed data inventory")
    parser.add_argument("--use-llm", action="store_true",
                        help="Use LLM to parse material names (caches results)")
    parser.add_argument("--llm-model", default="gemini-2.5-flash",
                        help="LLM model for material name parsing (default: gemini-2.5-flash)")
    parser.add_argument("--y-label", default=None,
                        help="Y-axis label for figures (default: auto-detect or 'Conversion (%%)')")
    parser.add_argument("--y-keywords", nargs="*", default=None,
                        help="Keywords to match y_axis_label in plot data (default: ['conversion'])")
    parser.add_argument("--ref-temp", type=float, default=500.0,
                        help="Reference temperature for interpolation (default: 500)")
    parser.add_argument("--skip-dirs", nargs="*", default=[],
                        help="Paper directories to skip")
    args = parser.parse_args()

    # ── Set global config from CLI ──
    DATA_DIR = args.data_dir.resolve()
    if not DATA_DIR.is_dir():
        parser.error(f"Data directory not found: {DATA_DIR}")

    if args.y_keywords:
        Y_KEYWORDS = [kw.lower() for kw in args.y_keywords]
    if args.y_label:
        Y_LABEL = args.y_label
    REF_TEMP = args.ref_temp
    if args.y_label:
        METRIC_NAME = args.y_label.replace("(%)", "").strip()
    skip_dirs = frozenset(args.skip_dirs)

    print(f"Data directory: {DATA_DIR}")
    print(f"Metric: {Y_LABEL} | keywords: {Y_KEYWORDS} | ref temp: {REF_TEMP}°C")

    # ── Optionally use LLM for material name parsing ──
    mat_cache = None
    if args.use_llm:
        mat_cache = llm_parse_all_materials(
            data_dir=DATA_DIR, skip_dirs=skip_dirs,
            model_name=args.llm_model,
        )

    print("\nLoading data...")
    df_curves, df_synthesis = load_all_data(
        skip_dirs=skip_dirs, material_cache=mat_cache,
    )
    print(f"  {len(df_curves)} performance curves loaded")
    print(f"  {len(df_synthesis)} synthesis records loaded")

    if skip_dirs:
        print(f"  Skipped directories: {', '.join(skip_dirs)}")

    if args.debug:
        print_debug(df_curves, df_synthesis)

    print("\nGenerating 7 publication figures...\n")
    make_fig1(df_curves)
    make_fig2(df_curves)
    make_fig3(df_synthesis)
    make_fig4(df_curves, df_synthesis)
    make_fig5(df_curves, df_synthesis)
    make_fig6(df_curves)
    make_fig7(df_curves)

    print("\nExporting data files...\n")
    export_landscape_csv(df_curves)

    print(f"\nAll outputs saved to {OUT_DIR}")
