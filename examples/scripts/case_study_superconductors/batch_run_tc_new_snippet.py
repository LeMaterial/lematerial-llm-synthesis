#!/usr/bin/env python3
"""
Batch Tc Extraction Script — Snippet-Enhanced + Synthesis
=============================================================
Runs the full superconductivity Tc extraction pipeline on every PDF in a folder.
Features:
  - Synthesis extraction (method, steps, evaluation)
  - Supports MULTI-CONDITION Tc from text (same material at different 
    pressures, etc.)
  - Runs DUAL VLM Tc extraction: original (single image) + snippet (full + 
    bottom-left crop)

Usage:
    python batch_run_tc_new_snippet.py /path/to/pdf_folder
    python batch_run_tc_new_snippet.py /path/to/pdf_folder --max 5
    python batch_run_tc_new_snippet.py /path/to/pdf_folder --skip-existing
    python batch_run_tc_new_snippet.py /path/to/pdf_folder --skip-figures

Results are saved to <pdf_folder>/results_snippet/<paper_id>/ and appended to
<pdf_folder>/results_snippet/tc_master_snippet.csv.
"""

import argparse
import base64
import csv
import io
import json
import os
import random
import re
import ssl
import sys
import time
import traceback
import warnings
from pathlib import Path

# ── Fix SSL for uv-managed Python on macOS ──
_ssl_cert = ssl.get_default_verify_paths().cafile
if _ssl_cert and os.path.exists(_ssl_cert):
    os.environ.setdefault("SSL_CERT_FILE", _ssl_cert)
    os.environ.setdefault("SSL_CERT_DIR", os.path.dirname(_ssl_cert))

# ── Add src to path ──
SRC_PATH = Path(__file__).resolve().parent.parent.parent / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(
    Path(__file__).resolve().parent.parent.parent / ".env", override=True
)

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
import logging  # noqa: E402

logging.getLogger("pydantic").setLevel(logging.ERROR)
logging.getLogger("LiteLLM").setLevel(logging.ERROR)
logging.getLogger("litellm").setLevel(logging.ERROR)

# ── Model config ──
GEMINI_MODEL = "gemini-3.0-flash"
CLAUDE_MODEL = "claude-sonnet-4-6"
LINKER_MODEL = "gemini-3.0-flash"


# =============================================================================
# HELPERS
# =============================================================================

SI_PATTERNS = [
    "_SI",
    "-SI",
    "_si",
    "-si",
    "_Supporting",
    "_supporting",
    "_Supplementary",
    "_supplementary",
    "_supp",
    "_Supp",
]


def find_si_file(main_paper_path: Path) -> Path | None:
    parent_dir = main_paper_path.parent
    main_stem = main_paper_path.stem
    for pattern in SI_PATTERNS:
        for ext in [".pdf", ".md", ".txt"]:
            si_path = parent_dir / f"{main_stem}{pattern}{ext}"
            if si_path.exists():
                return si_path
    return None


def load_file_text(path: Path, pdf_extractor=None) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        if pdf_extractor is None:
            from llm_synthesis.transformers.pdf_extraction import (
                MistralPDFExtractor,
            )

            pdf_extractor = MistralPDFExtractor(structured=False)
        with open(path, "rb") as f:
            return pdf_extractor.forward(f.read())
    elif suffix in [".md", ".txt"]:
        with open(path, errors="replace") as f:
            return f.read()
    else:
        raise ValueError(f"Unsupported file type: {suffix}")


def fallback_check_rt_plot(plot, fig) -> bool:
    context = (
        f"{fig.context_before or ''} "
        f"{fig.context_after or ''} "
        f"{fig.alt_text or ''}"
    ).lower()
    rt_context_hints = [
        "resistivity",
        "resistance",
        "ρ(t)",
        "ρ vs",
        "r(t)",
        "r vs t",
        "t (k)",
        "t [k]",
        "temperature dependence of ρ",
        "temperature dependence of the resistivity",
        "temperature dependence of the resistance",
        "μω cm",
        "μω·cm",
        "mω cm",
    ]
    return any(hint in context for hint in rt_context_hints)


def _is_axis_missing(label, unit) -> bool:
    return (not label or not label.strip()) and (not unit or not unit.strip())


# =============================================================================
# MULTI-CONDITION Tc TEXT PARSING
# =============================================================================


def parse_tc_text_response_multi(raw_text: str) -> list[dict]:
    """Parse multi-condition Tc text extraction into a list of dicts.

    Each dict: {material, condition, superconducting, T_onset, Tc_mid,
    T_zero, delta_Tc}

    Handles labeled format (preferred):
        Fe1.03Te0.63Se0.37 | condition: single-crystal |
        superconducting: YES | T_onset: 14.8 K | Tc: NR |
        T_zero: NR | delta_Tc: 1.3 K
    And unlabeled fallback:
        Fe1.03Te0.63Se0.37 | single-crystal | YES | 14.8 K | NR | NR | 1.3 K

    Post-processing: if Tc_mid is missing but T_onset and delta_Tc
    are both present, computes Tc_mid ≈ T_onset − delta_Tc / 2.
    """
    # Regex to detect condition-like strings (contain x=, H=, GPa, etc.)
    _condition_re = re.compile(
        r"(x\s*[=:]|h\s*[=:]\s*\d|p\s*[=:]\s*\d|gpa|kbar|mpa|"
        r"single.?crystal|poly|thin.?film|bulk|"
        r"ambient|as.?grown|anneal|dop|optimally|undoped)",
        re.IGNORECASE,
    )

    # All Tc-like label keys we recognise
    _tc_label_re = r"(t_onset|tc_mid|tc|t_zero|delta_tc|δtc|Δtc)"

    results = []
    for line in raw_text.strip().split("\n"):
        line = line.strip()
        if not line or "|" not in line:
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 2:
            continue
        material = parts[0].strip()
        if not material or material.lower().startswith("material"):
            continue  # skip header lines

        entry = {
            "material": material,
            "condition": "ambient",
            "superconducting": False,
            "T_onset": None,
            "Tc_mid": None,
            "T_zero": None,
            "delta_Tc": None,
        }

        remaining = parts[1:]

        # Detect if format is labeled
        # (has "condition:", "superconducting:", "Tc:", etc.)
        has_labels = any(
            re.match(
                r"(condition|superconducting|t_onset|tc_mid|tc|t_zero|delta_tc|δtc|Δtc)\s*:",
                p.strip().lower(),
            )
            for p in remaining
        )

        def _assign_tc_key(key: str, val: float):
            """Assign a parsed Tc value to the right entry field."""
            if key in ("tc", "tc_mid"):
                entry["Tc_mid"] = val
            elif key == "t_onset":
                entry["T_onset"] = val
            elif key == "t_zero":
                entry["T_zero"] = val
            elif key in ("delta_tc", "δtc", "Δtc"):
                entry["delta_Tc"] = val

        if has_labels:
            # ── Labeled format ──
            for part in remaining:
                pl = part.strip().lower()
                if pl.startswith("condition:"):
                    entry["condition"] = (
                        part.strip().split(":", 1)[1].strip() or "ambient"
                    )
                elif "superconducting" in pl:
                    entry["superconducting"] = "yes" in pl
                else:
                    match = re.match(
                        _tc_label_re + r"\s*:\s*(\d+\.?\d*)\s*k?", pl
                    )
                    if match:
                        _assign_tc_key(match.group(1), float(match.group(2)))
        else:
            # ── Unlabeled fallback ──
            bare_temps = []

            for part in remaining:
                ps = part.strip()
                pl = ps.lower()

                # YES/NO
                if pl in ("yes", "no"):
                    entry["superconducting"] = pl == "yes"
                    continue

                # NR / N/A
                if pl in ("nr", "n/a", "none", "-", "—", ""):
                    continue

                # Inline labeled value (e.g. "Tc: 16 K", "delta_Tc: 1.3 K")
                lm = re.match(_tc_label_re + r"\s*[:=]\s*(\d+\.?\d*)\s*k?", pl)
                if lm:
                    _assign_tc_key(lm.group(1), float(lm.group(2)))
                    continue

                # Condition-like string BEFORE bare number check
                if _condition_re.search(pl):
                    if entry["condition"] == "ambient":
                        entry["condition"] = ps
                    continue

                # Bare number with K unit → temperature
                vm = re.match(r"^\s*(\d+\.?\d*)\s*k\s*$", pl)
                if vm:
                    bare_temps.append(float(vm.group(1)))
                    continue

                # Bare number without unit
                vm2 = re.match(r"^\s*(\d+\.?\d*)\s*$", pl)
                if vm2:
                    bare_temps.append(float(vm2.group(1)))
                    continue

                # Anything else → condition
                if entry["condition"] == "ambient":
                    entry["condition"] = ps

            # Assign positional temps: T_onset, Tc_mid, T_zero
            for i, val in enumerate(bare_temps):
                if i == 0:
                    entry["T_onset"] = val
                elif i == 1:
                    entry["Tc_mid"] = val
                elif i == 2:
                    entry["T_zero"] = val

        # ── Post-processing: compute Tc_mid from onset + delta if missing ──
        if (
            entry["Tc_mid"] is None
            and entry["T_onset"] is not None
            and entry["delta_Tc"] is not None
        ):
            entry["Tc_mid"] = round(entry["T_onset"] - entry["delta_Tc"] / 2, 2)

        results.append(entry)
    return results


def _normalize_formula(s: str) -> str:
    def _is_annotation(paren_content: str) -> bool:
        """Return True if parenthesized content is an annotation,
        not formula.
        """
        c = paren_content.strip()
        # Sample labels: S1, SC1, #1, etc.
        if re.match(r"^(?:S|SC|#)\d+$", c, re.IGNORECASE):
            return True
        # Annotations with keywords or doping labels
        if re.search(
            r"x\s*=|sintered|single|crystal|ambient|pristine|bulk|film",
            c,
            re.IGNORECASE,
        ):
            return True
        # Compositional: contains element symbols and numbers, possibly commas
        if re.match(r"^[A-Za-z0-9.,\s_\-]+$", c) and re.search(
            r"[A-Z][a-z]?", c
        ):
            return False  # looks like formula content
        return True  # default: treat as annotation

    # Strip ALL parenthesized annotations
    # (anywhere in string, not just trailing)
    # Process right-to-left so positions don't shift
    base = s.strip()
    while True:
        changed = False
        for m in reversed(list(re.finditer(r"\s*\(([^)]*)\)", base))):
            if _is_annotation(m.group(1)):
                base = base[: m.start()] + base[m.end() :]
                changed = True
                break  # restart after modification
        if not changed:
            break
    # Clean measurement annotations (ρ_ab, ρ_c, etc.)
    base = re.sub(
        r"[\s_]*[ρrho]+[\s_]*(?:ab|c|xx|yy|zz)\b", "", base, flags=re.IGNORECASE
    )
    base = re.sub(r"[\s_]+$", "", base)

    # Strip compositional parentheses WITHOUT commas:
    # Fe1.03(Te0.63Se0.37) -> Fe1.03Te0.63Se0.37
    # Keep parentheses WITH commas: (Cu,C) stays as-is (substitution notation)
    def _strip_simple_parens(m):
        content = m.group(1)
        if "," in content:
            return m.group(0)  # keep (Cu,C) etc.
        return content

    base = re.sub(r"\(([^)]+)\)", _strip_simple_parens, base)
    # Strip LaTeX subscript/superscript braces:
    # _{...} -> content, ^{...} -> content
    base = re.sub(r"_\{([^}]*)\}", r"\1", base)
    base = re.sub(r"\^\{([^}]*)\}", r"\1", base)
    base = re.sub(r"\{([^}]*)\}", r"\1", base)
    base = base.replace("δ", "delta").replace("Δ", "delta")
    base = (
        base.replace("₀", "0")
        .replace("₁", "1")
        .replace("₂", "2")
        .replace("₃", "3")
    )
    base = (
        base.replace("₄", "4")
        .replace("₅", "5")
        .replace("₆", "6")
        .replace("₇", "7")
    )
    base = base.replace("₈", "8").replace("₉", "9").replace("₋", "-")
    base = base.lower().replace(" ", "").replace("−", "-")
    # Strip trailing zeros from decimal numbers: 0.80 -> 0.8, 0.10 -> 0.1
    base = re.sub(r"(\.\d*?)0+(?=\D|$)", r"\1", base)
    return base


# All element symbols for material name validation
_ELEMENT_SYMBOLS = {
    "He",
    "Li",
    "Be",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "Cl",
    "Ar",
    "Ca",
    "Sc",
    "Ti",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
    "Rb",
    "Sr",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Sb",
    "Te",
    "Xe",
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Po",
    "At",
    "Rn",
    "Fr",
    "Ra",
    "Ac",
    "Th",
    "Pa",
    "Np",
    "Pu",
    "Am",
    "Cm",
    "Bk",
    "Cf",
    "Es",
    "Fm",
    "Md",
    "No",
    "Lr",
    "Rf",
    "Db",
    "Sg",
    "Bh",
    "Hs",
    "Mt",
    "Ds",
    "Rg",
    "Cn",
    "Nh",
    "Fl",
    "Mc",
    "Lv",
    "Ts",
    "Og",
    "W",
    "U",
    "H",
    "B",
    "C",
    "N",
    "O",
    "F",
    "P",
    "S",
    "K",
    "V",
    "Y",
    "I",
}


def is_valid_material_name(name: str) -> bool:
    """Check if a string looks like a valid chemical/material formula.

    Rejects measurement parameters (fields, pressures, currents),
    bare doping labels, irradiation doses, and color annotations.
    Returns True only if the name contains at least one element symbol.
    """
    stripped = name.strip()
    if not stripped or len(stripped) < 2:
        return False

    # --- NEGATIVE filters: reject known non-material patterns ---

    # Magnetic field values: "5 T", "0.5 kOe", "100 Oe", "50 mT"
    if re.match(
        r"^[\d.,\s]*\d+\.?\d*\s*(T|kOe|Oe|mT)$", stripped, re.IGNORECASE
    ):
        return False

    # Pressure values: "P_300K = 0.65 GPa", "2.5 GPa", "10 kbar"
    if re.search(r"(GPa|MPa|kbar)\b", stripped, re.IGNORECASE):
        return False
    if re.match(r"^P[\d₀₁₂₃₄₅₆₇₈₉]*K?\s*[=:]", stripped):
        return False

    # Current values: "100 µA", "5 mA", "10 nA"
    if re.match(
        r"^[\d.,\s]*\d+\.?\d*\s*(µA|uA|mA|nA|A)\s*$", stripped, re.IGNORECASE
    ):
        return False

    # Bare doping labels: "x = 0.1", "x=0.89 (blue)", "x=0.23_C"
    if re.match(r"^x\s*[=:]\s*[\d.]", stripped, re.IGNORECASE):
        return False

    # Irradiation doses: "2*10^16 p/cm^2", "6.4 × 10¹⁶ p/cm²", "anneal"
    if re.search(r"\d+\s*[*×x]\s*10[\^⁰¹²³⁴⁵⁶⁷⁸⁹]", stripped):
        return False
    if re.search(r"(p/cm|n/cm|anneal|irradiat)", stripped, re.IGNORECASE):
        return False

    # Temperature values: "300 K", "T = 4.2 K"
    if re.match(r"^T?\s*[=:]?\s*[\d.]+\s*K\s*$", stripped, re.IGNORECASE):
        return False

    # Names that are ONLY numbers, symbols, and operators
    if re.match(r"^[\d\s.,+\-*/^()=%]+$", stripped):
        return False

    # Labels like "pristine (x=0.107)", "as-grown", "sample A"
    if re.match(
        r"^(pristine|as[- ]grown|as[- ]sintered|undoped|sample)\b",
        stripped,
        re.IGNORECASE,
    ):
        return False

    # --- POSITIVE check: must contain at least one element symbol ---
    # Remove trailing parenthetical annotations like "(blue)", "(red circles)"
    text_to_check = re.sub(r"\s*\([^)]*\)\s*$", "", stripped).strip()
    if not text_to_check:
        return False

    # Check for two-letter element symbols first (avoid substring issues)
    for symbol in sorted(_ELEMENT_SYMBOLS, key=len, reverse=True):
        if len(symbol) == 2:
            # Two-letter: match Xx pattern not surrounded by other letters
            if re.search(
                r"(?<![a-zA-Z])" + re.escape(symbol) + r"(?![a-z])",
                text_to_check,
            ):
                return True
        else:
            # Single-letter: must be uppercase and not part of a longer word
            if re.search(
                r"(?<![a-zA-Z])" + re.escape(symbol) + r"(?![a-z])",
                text_to_check,
            ):
                return True

    return False


def smart_split_materials(text: str) -> list[str]:
    """Split a comma/newline-separated material list, respecting parentheses.

    '(Cu,C)Ba2Ca2Cu3O9' will NOT be split at the internal comma.
    """
    text = text.replace("\n", ",")
    materials = []
    current = []
    depth = 0

    for char in text:
        if char == "(":
            depth += 1
            current.append(char)
        elif char == ")":
            depth = max(0, depth - 1)
            current.append(char)
        elif char == "," and depth == 0:
            token = "".join(current).strip()
            if token:
                materials.append(token)
            current = []
        else:
            current.append(char)

    # Don't forget the last token
    token = "".join(current).strip()
    if token:
        materials.append(token)

    return [m for m in materials if m]


def find_matching_text_tc_multi(
    material: str, tc_entries: list[dict]
) -> list[dict]:
    """Return all multi-condition entries whose material fuzzy-matches."""
    mat_norm = _normalize_formula(material)
    matches = []
    for entry in tc_entries:
        key_norm = _normalize_formula(entry["material"])
        if key_norm == mat_norm:
            matches.append(entry)
    matches.sort(
        key=lambda x: (
            not x.get("superconducting", False),
            -(x.get("Tc_mid") or 0),
        )
    )
    return matches


# =============================================================================
# VLM Tc PROMPTS + PARSING
# =============================================================================

# ── PROMPT A: Original (single image) ──
PROMPT_TEMPLATE_ORIGINAL = """
You are analyzing a Resistance (or Resistivity) vs Temperature plot from a
superconductivity paper. Your task is to determine the critical temperature
Tc for each series using the standard geometric construction.

CRITICAL DISTINCTION — SUPERCONDUCTING TRANSITION vs NORMAL METALLIC BEHAVIOR:
Many materials (especially heavy-fermion compounds like CeCoIn5, CeRhIn5, etc.)
show a GRADUAL decrease in resistivity over a WIDE temperature range (e.g., from
50 K down to 5 K). This is NORMAL metallic behavior (Kondo coherence, phonon
scattering reduction, etc.) and is NOT a superconducting transition.

The superconducting transition has these characteristics:
  - It is a SHARP, near-vertical drop in resistance
  - It occurs over a NARROW temperature range (typically 0.1 to 3 K wide)
  - Resistance drops from a finite value all the way to ZERO
    (or very close to zero)
  - It looks like a cliff or step function, not a gradual slope

If you see a curve that gradually decreases over 10-50 K, that is NOT
superconductivity — it is normal metallic/Kondo behavior.

STEP 0 — EXAMINE THE FULL FIGURE (main plot + any insets/panels):
a) Identify ALL panels in the figure: the main plot and any insets, secondary
   panels, or embedded sub-plots. For each one, describe:
     - What quantity is on each axis
       (e.g., rho vs T, Tc vs x, phase diagram, etc.)
     - The axis ranges and tick marks
     - Whether it contains information relevant to determining Tc

b) CATEGORIZE each inset/panel into one of these types:
     (i)   ZOOMED R(T): A magnified view of the transition region.
           -> Use this PREFERENTIALLY for geometric Tc construction.
     (ii)  Tc SUMMARY: Shows Tc vs composition/pressure/doping/field.
           -> Read Tc values DIRECTLY from this panel.
     (iii) OTHER: Not useful for Tc determination.

c) Read ALL numbered tick marks on the main plot axes.

d) CRITICAL — SCALE AWARENESS: If the temperature axis spans a wide range
   (e.g., 0-300 K) and the transitions happen in a small fraction, prefer
   reading from a zoomed inset if available.

{figure_caption_block}

STEP 0.5 — EXTRACT Tc FROM SUMMARY INSETS (if any type-(ii) inset found):
Read Tc values directly from it:
  inset_tc_<series_name>: <value> K
These serve as a REFERENCE for Step 4 cross-check.

{materials_context_block}

STEP 1 — IDENTIFY SERIES:
{series_name_instruction}
List every distinct curve visible in the plot.

STEP 2 — READ RESISTANCE VALUES AT LOWEST AND HIGHEST TEMPERATURE:
For EACH series read:
  a) R_at_lowest_T: resistance at the LOWEST temperature shown
  b) R_at_highest_T: resistance at the HIGHEST temperature shown

STEP 3 — CONFIRM SUPERCONDUCTIVITY:
Superconducting ONLY if R_at_lowest_T is approximately zero.

STEP 4 — GEOMETRIC Tc CONSTRUCTION (only for confirmed superconductors):
Use a zoomed inset if available, otherwise the main plot.

IMPORTANT — SCAN FROM HIGH T TO LOW T (right to left on the plot):

  a) R_normal: Starting from the HIGH-temperature side, read the resistance
     on the plateau IMMEDIATELY BEFORE the sharp superconducting drop.
     NOT the maximum resistance at 300 K — the value just above the drop.

  b) T_onset: Scanning LEFT from R_normal, the temperature where resistance
     FIRST begins to drop sharply. This is the HIGH-temperature edge.

  c) T_zero: Continue scanning LEFT. The temperature where resistance FIRST
     reaches approximately zero. NOT the lowest temperature in the dataset.

     EXAMPLE: If R~0 from 2 K to 25 K, then T_zero = 25 K (NOT 2 K).
     T_zero is where the flat R~0 region ENDS going to higher T.

  d) SANITY CHECK: T_onset - T_zero should be 0.5-5 K. If >10 K, re-examine.

  e) Tc_mid = (T_onset + T_zero) / 2.  Should be in the MIDDLE of the sharp
     drop, not at the left edge of the data.

  f) Delta_Tc = T_onset - T_zero

  g) CROSS-CHECK: If inset Tc differs by >20%, use inset value.

COMMON MISTAKE: Do NOT confuse the lowest data-point temperature (e.g., 2 K)
with T_zero. If R=0 from 2 K up to 25 K, T_zero = 25 K.

STEP 5 — RELATIVE ORDERING OF TRANSITIONS:
Compare which series transitions at higher vs lower T.

Output format:

inset_detected: <yes/no>
inset_type: <"zoomed_rt" | "tc_summary" | "other" | "none">
inset_description: <brief description or "N/A">
inset_axes: <tick marks if present, otherwise "N/A">
inset_tc_values: <series: value K, ... or "N/A">
main_x_axis_ticks: <list>
main_y_axis_ticks: <list>

Series: <name>
R_at_lowest_T: <value and unit>
R_at_highest_T: <value and unit>
superconducting: <YES/NO>
[If NO:]
reason: <why not superconducting>
[If YES:]
R_normal: <value and unit>
T_onset: <value> K
T_zero: <value> K
Tc_mid: <value> K
Delta_Tc: <value> K
source: <"inset" or "main plot" or "zoomed inset">

relative_ordering: <which series transitions first, etc.>

Do not output any other text.
"""


# ── PROMPT B: Snippet-enhanced (two images) ──
PROMPT_TEMPLATE_SNIPPET = """
You are analyzing a Resistance (or Resistivity) vs Temperature plot from a
superconductivity paper. You have TWO views of the same figure:
  - Image 1: The FULL plot (complete axes, legend, all series)
  - Image 2: A ZOOMED crop of the BOTTOM-LEFT quadrant (low temperature,
    low resistance region) — this has ~4x the pixel resolution in the
    transition region where superconducting drops happen.

Your task is to determine the critical temperature Tc for each series using
the standard geometric construction.

CRITICAL DISTINCTION — SUPERCONDUCTING TRANSITION vs NORMAL METALLIC BEHAVIOR:
Many materials (especially heavy-fermion compounds like CeCoIn5, CeRhIn5, etc.)
show a GRADUAL decrease in resistivity over a WIDE temperature range (e.g., from
50 K down to 5 K). This is NORMAL metallic behavior (Kondo coherence, phonon
scattering reduction, etc.) and is NOT a superconducting transition.

The superconducting transition has these characteristics:
  - It is a SHARP, near-vertical drop in resistance
  - It occurs over a NARROW temperature range (typically 0.1 to 3 K wide)
  - Resistance drops from a finite value all the way to ZERO
    (or very close to zero)
  - It looks like a cliff or step function, not a gradual slope

STEP 0 — EXAMINE BOTH IMAGES:
a) From Image 1 (full plot): Identify ALL panels, insets, legend entries,
   axis labels, axis units, and axis ranges. Read ALL numbered tick marks.

b) From Image 2 (bottom-left crop): Read the tick marks visible in this
   zoomed view. These are CRITICAL for precise Tc determination. Note:
   - What temperature range is visible?
   - What resistance range is visible?
   - Can you see individual data points or transitions more clearly?

   CROP VALIDATION: Verify that Image 2 actually shows the bottom-left
   region of Image 1. The crop's x-axis should start at the same minimum
   temperature as Image 1, and its y-axis should start at the same minimum
   resistance. If the crop does NOT match (e.g., wrong axis labels, wrong
   scale, or it shows a different panel/inset), IGNORE Image 2 entirely
   and use only Image 1 for all analysis.

c) CATEGORIZE each inset/panel into:
     (i)   ZOOMED R(T) -> Use PREFERENTIALLY for geometric Tc construction.
     (ii)  Tc SUMMARY  -> Read Tc values DIRECTLY.
     (iii) OTHER        -> Note but do not use.

d) CRITICAL — SCALE AWARENESS: Compare tick marks from Image 1 vs Image 2.
   If the full plot spans 0-300 K but the crop shows 0-150 K with clear tick
   marks, USE THE CROP for reading transition temperatures. If the transition
   falls outside the crop, use the full image.

{figure_caption_block}

STEP 0.5 — EXTRACT Tc FROM SUMMARY INSETS (if any type-(ii) inset found):
Read Tc values directly:
  inset_tc_<series_name>: <value> K
These serve as a REFERENCE for Step 4 cross-check.

{materials_context_block}

STEP 1 — IDENTIFY SERIES:
{series_name_instruction}
List every distinct curve visible in the plot.

STEP 2 — READ RESISTANCE VALUES AT LOWEST AND HIGHEST TEMPERATURE:
For EACH series, use Image 2 (crop) for low-T values if the series is
visible there:
  a) R_at_lowest_T: resistance at the LOWEST temperature
  b) R_at_highest_T: resistance at the HIGHEST temperature

STEP 3 — CONFIRM SUPERCONDUCTIVITY:
Superconducting ONLY if R_at_lowest_T is approximately zero.

STRICT ZERO CHECK (use Image 2 crop for better precision):
In the cropped view, check whether the series data points at low T are
truly sitting ON the y=0 line. With the higher resolution of the crop,
you can distinguish between R~0 (touching the x-axis) and R that is merely
small but nonzero (floating above the x-axis). If a series has R clearly
above zero in the crop — even if it looked close to zero in the full image
— it is NOT superconducting.

STEP 4 — GEOMETRIC Tc CONSTRUCTION (only for confirmed superconductors):
Use the best view — in order of preference:
  1. A zoomed inset (if present in the figure)
  2. Image 2 (bottom-left crop) — if the transition is visible there
  3. Image 1 (full plot) — only if the transition is outside the crop

IMPORTANT — SCAN FROM HIGH T TO LOW T (right to left on the plot):

  a) R_normal: Starting from the HIGH-temperature side, read the resistance
     value on the plateau IMMEDIATELY BEFORE the sharp superconducting drop.
     This is the normal-state resistance just above the transition — NOT the
     maximum resistance at 300 K.

  b) T_onset: Scanning LEFT from R_normal, find the temperature where the
     resistance FIRST begins to drop sharply below the normal-state plateau.
     This is the HIGH-temperature edge of the transition.

  c) T_zero: Continue scanning LEFT. T_zero is the temperature where the
     resistance FIRST reaches approximately zero. This is the LOW-temperature
     edge of the transition. T_zero is NOT the lowest temperature in the
     dataset — it is the temperature where R first hits zero.

     EXAMPLE: If data points are at R~0 from 2 K to 25 K, and then R rises
     sharply above 25 K, then T_zero = 25 K (NOT 2 K). T_zero is where the
     flat R~0 region ENDS going right (i.e., where the transition begins
     going from low T upward).

  d) SANITY CHECK: T_onset - T_zero should typically be 0.5-5 K for most
     superconductors. If > 10 K, you are probably measuring metallic decline,
     NOT the SC transition. Re-examine.

  e) Tc_mid = (T_onset + T_zero) / 2.   This should be in the MIDDLE of the
     sharp drop, not at the left edge of the data.

  f) Delta_Tc = T_onset - T_zero

  g) CROSS-CHECK: If inset Tc differs by >20%, use inset value.

COMMON MISTAKE TO AVOID: Do NOT confuse the lowest temperature data point
(e.g., 2 K) with T_zero. If R is zero from 2 K all the way up to 25 K,
then T_zero = 25 K. The VLM must identify WHERE the drop happens, not
where the data starts.

STEP 5 — RELATIVE ORDERING OF TRANSITIONS:
Compare which series transitions at higher vs lower T.

Output format:

inset_detected: <yes/no>
inset_type: <"zoomed_rt" | "tc_summary" | "other" | "none">
inset_description: <brief description or "N/A">
inset_axes: <tick marks if present, otherwise "N/A">
inset_tc_values: <series: value K, ... or "N/A">
crop_x_range: <temperature range visible in Image 2, e.g., "0 to 150 K">
crop_y_range: <resistance range visible in Image 2>
main_x_axis_ticks: <list>
main_y_axis_ticks: <list>

Series: <name>
R_at_lowest_T: <value and unit>
R_at_highest_T: <value and unit>
superconducting: <YES/NO>
[If NO:]
reason: <why not superconducting>
[If YES:]
R_normal: <value and unit>
T_onset: <value> K
T_zero: <value> K
Tc_mid: <value> K
Delta_Tc: <value> K
source: <"inset" or "main plot" or "zoomed inset" or "bottom-left crop">

relative_ordering: <which series transitions first, etc.>

Do not output any other text.
"""


# ── Image helpers ──


def crop_bottom_left_quadrant(img_base64: str) -> str:
    """Crop bottom-left quadrant -> base64 PNG.

    ~4x resolution in SC transition region.
    """
    from PIL import Image as PILImage

    img_bytes = base64.b64decode(img_base64)
    img = PILImage.open(io.BytesIO(img_bytes))
    w, h = img.size
    cropped = img.crop((0, h // 2, w // 2, h))
    buf = io.BytesIO()
    cropped.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def extract_figure_caption(
    fig_ref: str, paper_text: str, max_chars: int = 500
) -> str:
    """Extract the caption for a figure from the paper text."""
    if not fig_ref or not paper_text:
        return ""
    ref_match = re.search(r"(\d+)", fig_ref)
    if not ref_match:
        return ""
    fig_num = ref_match.group(1)

    pat = (
        r"(?:Fig(?:ure)?\.?\s*"
        + fig_num
        + r")\s*[.:)\s]\s*"
        + r"(.*?)"
        + r"(?:\n\n|\n(?:Fig|FIG|TABLE|\\)|$)"
    )
    match = re.search(pat, paper_text, re.IGNORECASE | re.DOTALL)
    if match:
        caption = re.sub(r"\s+", " ", match.group(1).strip())
        return caption[:max_chars]

    simple = re.search(
        r"(?:Fig(?:ure)?\.?\s*" + fig_num + r")\s*[.:)]\s*(.{10,300})",
        paper_text,
        re.IGNORECASE,
    )
    if simple:
        return re.sub(r"\s+", " ", simple.group(1).strip())[:max_chars]
    return ""


def build_tc_prompt(
    known_series_names: list[str] | None = None,
    use_snippet: bool = False,
    figure_caption: str = "",
    materials: list[str] | None = None,
) -> str:
    """Build the Tc extraction prompt (original or snippet).

    Args:
        known_series_names: Series names from plot extraction (legend labels).
        use_snippet: If True, use the two-image prompt.
        figure_caption: Figure caption text from the paper.
        materials: List of material compositions from Step 1 extraction.
    """
    template = (
        PROMPT_TEMPLATE_SNIPPET if use_snippet else PROMPT_TEMPLATE_ORIGINAL
    )

    # Build series naming instruction — tell VLM to map to real formulas
    if known_series_names and len(known_series_names) > 0:
        names_list = "\n".join(
            f"  {i + 1}. {name}" for i, name in enumerate(known_series_names)
        )
        instruction = (
            "The following series labels were found in this plot's legend:\n"
            f"{names_list}\n\n"
            "These are plot legend labels which may be abbreviated "
            "(e.g., 'x=0.12', '5 T', '0 GPa'). You MUST map each "
            "curve to its actual material composition using the "
            "MATERIALS CONTEXT below and the figure caption. Always "
            "output full chemical formulas as series names "
            "(e.g., 'Re0.88Mo0.12'), NOT bare labels like 'x=0.12' "
            "or condition labels like '5 T'.\n\n"
            "If you see additional curves not in this list, add them "
            "with their full chemical formula."
        )
    else:
        instruction = (
            "List every distinct curve by its material composition "
            "(chemical formula)."
        )

    # First pass: preserve the inner placeholders
    prompt = template.format(
        series_name_instruction=instruction,
        figure_caption_block="{figure_caption_block}",
        materials_context_block="{materials_context_block}",
    )

    # Inject materials context block
    if materials and len(materials) > 0:
        mat_list = "\n".join(f"  {i + 1}. {m}" for i, m in enumerate(materials))
        materials_block = (
            f"MATERIALS CONTEXT FROM THIS PAPER:\n"
            f"The following materials were identified in this paper:\n"
            f"{mat_list}\n\n"
            "Use this list as CONTEXT to understand what compound "
            "family is being studied.\n"
            "When the plot legend uses abbreviated labels like "
            "'x=0.12', 'x=0.20', etc.,\n"
            "you should RESOLVE these to full chemical formulas. "
            "For example:\n"
            f"  - If the paper studies Re_{{1-x}}Mo_x and the legend "
            f"says 'x=0.12',\n"
            "    the series name should be 'Re0.88Mo0.12'\n"
            f"  - If the paper studies Ta2Pd_{{1-x}}S_x and the "
            "caption shows x=0.1, x=0.2,\n"
            "    the series names should be "
            "'Ta2Pd0.9S0.1', 'Ta2Pd0.8S0.2'\n\n"
            "You are NOT restricted to only materials in this list. "
            "If the plot shows\n"
            "compositions not listed above (e.g., additional doping "
            "levels), resolve\n"
            "them to full chemical formulas using the same "
            "pattern.\n\n"
            "IMPORTANT — EXCLUDE NON-COMPOSITION SERIES:\n"
            "If multiple curves represent the SAME material under "
            "different measurement\n"
            "conditions (different magnetic fields like 0 T, 1 T, "
            "5 T; different pressures\n"
            "like 0 GPa, 1.5 GPa; different currents like "
            "100 uA, 10 mA), these are\n"
            "NOT different materials. Report ONLY the zero-field / "
            "ambient / lowest-\n"
            "excitation curve for Tc determination. Skip the "
            "field/pressure/current-\n"
            "dependent curves entirely.\n\n"
            "IMPORTANT — SERIES NAMING:\n"
            "Do NOT append measurement annotations to series names. "
            "If the plot shows\n"
            "resistivity in different directions "
            "(rho_ab, rho_c, ρ_ab, ρ_c) or labels\n"
            "like (SC1), (SC2), (#1), (#2), these describe HOW "
            "something was measured,\n"
            "not WHAT the material is. The series name should be "
            "ONLY the chemical\n"
            "formula (e.g., 'Fe1.03Te0.63Se0.37', NOT "
            "'Fe1.03Te0.63Se0.37_ρ_ab').\n"
            "If the same material has multiple curves for different "
            "measurement\n"
            "directions, pick the one that most clearly shows the "
            "Tc transition."
        )
    else:
        materials_block = ""
    prompt = prompt.replace("{materials_context_block}", materials_block)

    # Inject caption block
    if figure_caption:
        caption_block = (
            f"FIGURE CAPTION (from the paper):\n"
            f"  {figure_caption}\n\n"
            "Use this caption to understand which materials/samples "
            "are shown, doping levels, pressure conditions, etc. "
            "This helps you correctly identify and label each series."
        )
    else:
        caption_block = ""
    return prompt.replace("{figure_caption_block}", caption_block)


# ── VLM response parsing ──


def parse_direct_tc_response(response_text: str) -> dict:
    results = {}
    current = None
    inset_tc_raw = None
    for line in response_text.strip().split("\n"):
        line = line.strip()
        if line.lower().startswith("inset_tc_values:") and current is None:
            inset_tc_raw = line.split(":", 1)[1].strip()
            continue
        if line.startswith("Series:"):
            current = line.split(":", 1)[1].strip()
            results[current] = {}
        elif current and ":" in line:
            key, val = line.split(":", 1)
            key = key.strip().lower().replace(" ", "_")
            val = val.strip()
            if key == "superconducting":
                results[current][key] = val.upper().startswith("YES")
            elif key in ("reason", "source"):
                results[current][key] = val
            else:
                match = re.search(r"(\d+\.?\d*|\d*\.\d+)", val)
                if match:
                    try:
                        results[current][key] = float(match.group(1))
                    except ValueError:
                        results[current][key] = val
                elif "no transition" in val.lower() or "n/a" in val.lower():
                    results[current][key] = None
                else:
                    results[current][key] = val
    if inset_tc_raw and inset_tc_raw.lower() not in ("n/a", "none", ""):
        inset_pairs = _parse_inset_tc_values(inset_tc_raw, results)
        for series_name, tc_val in inset_pairs.items():
            if series_name in results:
                results[series_name]["inset_tc"] = tc_val
    return results


def _parse_inset_tc_values(raw: str, known_series: dict) -> dict:
    result = {}
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry or ":" not in entry:
            continue
        name_part, val_part = entry.rsplit(":", 1)
        name_part = name_part.strip()
        val_part = val_part.strip()
        match = re.search(r"(\d+\.?\d*|\d*\.\d+)", val_part)
        if match:
            tc_val = float(match.group(1))
            if name_part in known_series:
                result[name_part] = tc_val
            else:
                np_lower = name_part.lower().replace(" ", "")
                for ks in known_series:
                    ks_lower = ks.lower().replace(" ", "")
                    if (
                        np_lower == ks_lower
                        or np_lower in ks_lower
                        or ks_lower in np_lower
                    ):
                        result[ks] = tc_val
                        break
    return result


def sanity_check_delta_tc(vlm_results: dict) -> dict:
    corrected = {}
    for series_name, vals in vlm_results.items():
        corrected[series_name] = dict(vals)
        vlm_says_sc = vals.get("superconducting", False)
        if not vlm_says_sc:
            inset_tc = vals.get("inset_tc")
            if inset_tc is not None and inset_tc > 0:
                corrected[series_name]["superconducting"] = True
                corrected[series_name]["tc_mid"] = inset_tc
                corrected[series_name]["source"] = "inset"
                corrected[series_name]["_inset_override"] = (
                    f"VLM said SC=NO, but inset Tc={inset_tc:.1f} K"
                    f" found. Using inset value."
                )
            continue
        delta = vals.get("delta_tc")
        if delta is not None and delta > 10:
            inset_tc = vals.get("inset_tc")
            if inset_tc is not None and inset_tc > 0:
                corrected[series_name]["tc_mid"] = inset_tc
                corrected[series_name]["source"] = "inset"
                corrected[series_name]["_sanity_override"] = (
                    f"Delta_Tc={delta:.1f} K too wide. Using inset "
                    f"Tc={inset_tc:.1f} K."
                )
                for k in ("t_onset", "t_zero", "delta_tc"):
                    corrected[series_name].pop(k, None)
            else:
                corrected[series_name]["superconducting"] = False
                corrected[series_name]["_sanity_override"] = (
                    f"Delta_Tc={delta:.1f} K too wide. Likely metallic decline."
                )
                for k in ("tc_mid", "t_onset", "t_zero", "delta_tc"):
                    corrected[series_name].pop(k, None)
            continue
        tc_mid = vals.get("tc_mid")
        inset_tc = vals.get("inset_tc")
        if tc_mid is not None and inset_tc is not None and inset_tc > 0:
            relative_diff = abs(tc_mid - inset_tc) / inset_tc
            if relative_diff > 0.20:
                corrected[series_name]["tc_mid"] = inset_tc
                corrected[series_name]["source"] = "inset"
                corrected[series_name]["_inset_override"] = (
                    f"Geometric Tc_mid={tc_mid:.1f} K vs inset Tc="
                    f"{inset_tc:.1f} K "
                    f"({relative_diff * 100:.0f}% off). Using inset."
                )
    return corrected


# ── VLM series name cleanup ──


def clean_vlm_series_name(name: str) -> str:
    """Strip measurement annotations from VLM series names.

    Removes suffixes like _ρ_ab, _ρ_c, (SC1), (SC2), ρ_ab, ρ_c, (#1), (#2)
    that describe HOW something was measured, not WHAT the material is.
    """
    # Remove parenthesized sample labels: (SC1), (SC2), (#1), (#2), (S1), etc.
    name = re.sub(r"\s*\((?:SC|S|#)\d+\)\s*", " ", name)
    # Remove resistivity direction suffixes:
    # _ρ_ab, _ρ_c, ρ_ab, ρ_c, _rho_ab, etc.
    name = re.sub(
        r"[\s_]*[ρrho]+[\s_]*(?:ab|c|xx|yy|zz)\b", "", name, flags=re.IGNORECASE
    )
    # Remove trailing underscores and whitespace
    name = re.sub(r"[\s_]+$", "", name)
    return name.strip()


def clean_vlm_tc_results(vlm_results: dict) -> dict:
    """Clean all VLM Tc result keys and merge duplicates created by cleanup.

    When cleanup makes two keys identical (e.g., 'Fe1.03Te0.63Se0.37_ρ_ab' and
    'Fe1.03Te0.63Se0.37_ρ_c' both become 'Fe1.03Te0.63Se0.37'), keep the entry
    with the highest Tc (most clearly shows transition).
    """
    cleaned = {}
    for key, val in vlm_results.items():
        clean_key = clean_vlm_series_name(key)
        if clean_key in cleaned:
            # Keep the one with higher tc_mid (clearest transition)
            existing_tc = cleaned[clean_key].get("tc_mid")
            new_tc = val.get("tc_mid")
            if new_tc is not None and (
                existing_tc is None or new_tc > existing_tc
            ):
                cleaned[clean_key] = val
        else:
            cleaned[clean_key] = val
    return cleaned


# ── Fuzzy matching helpers ──


def _find_vlm_tc_for_series(
    series_name: str, vlm_results: dict, material_name: str | None = None
) -> dict:
    """Find VLM Tc data for a series, trying multiple matching strategies.

    Args:
        series_name: The plot-extraction series name (e.g., 'x = 0.12')
        vlm_results: Dict mapping VLM series names to Tc data
        material_name: Optional linked material name (e.g., 'NdO0.88F0.12FeAs')
    """
    # 1. Exact match on series_name
    if series_name in vlm_results:
        return vlm_results[series_name]
    # 2. Exact match on material_name (VLM now often returns full formulas)
    if material_name and material_name in vlm_results:
        return vlm_results[material_name]

    def _norm(s):
        s = (
            s.replace("₂", "2")
            .replace("₄", "4")
            .replace("₇", "7")
            .replace("₁", "1")
        )
        s = (
            s.replace("₃", "3")
            .replace("₅", "5")
            .replace("₆", "6")
            .replace("₈", "8")
        )
        s = (
            s.replace("₉", "9")
            .replace("₀", "0")
            .replace("₋", "-")
            .replace("δ", "delta")
        )
        s = re.sub(r"_\{([^}]*)\}", r"\1", s)
        s = re.sub(r"\^\{([^}]*)\}", r"\1", s)
        s = re.sub(r"\{([^}]*)\}", r"\1", s)
        return s.lower().replace(" ", "")

    sn = _norm(series_name)
    mn = _norm(material_name) if material_name else None

    for vlm_key, vlm_val in vlm_results.items():
        vk = _norm(vlm_key)
        # 3. Normalized substring match on series_name
        if sn in vk or vk in sn:
            return vlm_val
        # 4. Normalized match on material_name (handles LaTeX differences)
        if mn and (mn == vk or mn in vk or vk in mn):
            return vlm_val
        # 5. First-token match
        sn_parts = re.split(r"[\s_-]+", series_name.lower())
        vk_parts = re.split(r"[\s_-]+", vlm_key.lower())
        if (
            len(sn_parts) >= 1
            and len(vk_parts) >= 1
            and sn_parts[0] == vk_parts[0]
        ):
            return vlm_val

    # 6. Doping-value match: extract numbers after 'x=' or 'x ='
    #    from series_name
    #    and find VLM key containing the same doping value in the formula
    doping_match = re.search(r"x\s*=\s*([\d.]+)", series_name)
    if doping_match:
        x_val = doping_match.group(1)  # e.g., "0.12"
        for vlm_key, vlm_val in vlm_results.items():
            vk = _norm(vlm_key)
            # Check if this VLM key contains the doping value AND the
            # material_name's base formula (if available)
            if x_val in vk:
                # If we have a material name, verify same compound family
                if mn:
                    # Extract element letters from material_name
                    # to verify family
                    mn_letters = re.sub(r"[^a-z]", "", mn)
                    vk_letters = re.sub(r"[^a-z]", "", vk)
                    if mn_letters == vk_letters:
                        return vlm_val
                else:
                    return vlm_val

    # 7. Single-result fallback
    if len(vlm_results) == 1:
        return next(iter(vlm_results.values()))
    return {}


def _vlm_data_has_tc(vlm_data: dict) -> bool:
    return bool(
        vlm_data.get("superconducting") and vlm_data.get("tc_mid") is not None
    )


# ── CSV helpers ──


def extract_year_from_arxiv_id(paper_id: str) -> int | None:
    clean = paper_id.split("_")[0]
    clean = re.sub(r"v\d+$", "", clean)
    # New-style arXiv ID: YYMM.NNNNN (e.g., 2301.12345)
    match = re.match(r"^(\d{2})(\d{2})\.\d+$", clean)
    if match:
        yy = int(match.group(1))
        return 2000 + yy if yy < 50 else 1900 + yy
    # Old-style arXiv ID: YYMMNNN or YYMMNNNN (7-8 digits, no dot)
    match = re.match(r"^(\d{2})(\d{2})\d{3,4}$", clean)
    if match:
        yy = int(match.group(1))
        return 2000 + yy if yy < 50 else 1900 + yy
    # Fallback: interpret first 4 digits as year
    match = re.match(r"^(\d{4})", clean)
    if match:
        return int(match.group(1))
    return None


def normalize_formula_for_csv(s: str) -> str:
    # Strip LaTeX subscript/superscript braces:
    # _{...} -> content, ^{...} -> content
    s = re.sub(r"_\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\^\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\{([^}]*)\}", r"\1", s)
    s = (
        s.replace("₀", "0")
        .replace("₁", "1")
        .replace("₂", "2")
        .replace("₃", "3")
    )
    s = (
        s.replace("₄", "4")
        .replace("₅", "5")
        .replace("₆", "6")
        .replace("₇", "7")
    )
    s = (
        s.replace("₈", "8")
        .replace("₉", "9")
        .replace("₋", "-")
        .replace("δ", "delta")
    )
    return s.replace("−", "-").strip()


def pick_best_tc(text_tc, vlm_tc_orig, vlm_tc_snip, text_onset=None):
    """Pick the best Tc value. Prefer snippet VLM > original VLM > text."""
    if vlm_tc_snip is not None:
        return vlm_tc_snip, "vlm_snippet"
    if vlm_tc_orig is not None:
        return vlm_tc_orig, "vlm_original"
    if text_tc is not None:
        return text_tc, "text"
    if text_onset is not None:
        return text_onset, "text_onset"
    return None, "none"


def derive_is_superconductor(
    tc_text=None,
    tc_text_onset=None,
    tc_text_zero=None,
    vlm_orig_tc=None,
    vlm_snip_tc=None,
) -> bool | None:
    """Derive is_superconductor from actual Tc measurements.

    Returns True if ANY valid Tc measurement (Tc_mid, T_onset, T_zero)
    exists and > 0.
    Returns False if all values are 0 or absent.
    Returns None only if there is no data at all.
    """
    values = [tc_text, tc_text_onset, tc_text_zero, vlm_orig_tc, vlm_snip_tc]
    non_none = [v for v in values if v is not None]
    if not non_none:
        return None  # no data at all
    return any(v > 0 for v in non_none)


# =============================================================================
# CSV SCHEMA
# =============================================================================

CSV_COLUMNS = [
    "paper_id",
    "year",
    "material",
    "material_normalized",
    "condition",
    "is_superconductor",
    # Text Tc
    "tc_text",
    "tc_text_onset",
    "tc_text_zero",
    "tc_text_source",
    # VLM Tc — original (single image)
    "tc_vlm_orig",
    "tc_vlm_orig_onset",
    "tc_vlm_orig_zero",
    "tc_vlm_orig_source",
    # VLM Tc — snippet (full + bottom-left crop)
    "tc_vlm_snip",
    "tc_vlm_snip_onset",
    "tc_vlm_snip_zero",
    "tc_vlm_snip_source",
    "tc_vlm_source_plot",
    # Best Tc
    "tc_best",
    "tc_best_source",
    "has_text_tc",
    "has_vlm_tc_orig",
    "has_vlm_tc_snip",
    # Synthesis
    "has_synthesis",
    "synthesis_method",
    "synthesis_score",
]


# =============================================================================
# PIPELINE — one paper
# =============================================================================


def process_one_paper(
    pdf_path: Path, output_dir: Path, skip_figures: bool = False
):
    """Run the full Tc extraction pipeline (with synthesis) on a single PDF."""
    import anthropic

    from llm_synthesis.models.paper import Paper
    from llm_synthesis.transformers.material_extraction.dspy_extraction import (
        DspyTextExtractor,
        make_dspy_text_extractor_signature,
    )
    from llm_synthesis.transformers.pdf_extraction import MistralPDFExtractor
    from llm_synthesis.transformers.performance_linking.plot_filter import (
        PlotFilter,
    )
    from llm_synthesis.utils.dspy_utils import get_llm_from_name
    from llm_synthesis.utils.markdown_utils import clean_text
    from llm_synthesis.utils.performance_utils import (
        aggregate_all_materials_performance,
        sanitize_filename,
    )

    paper_id = pdf_path.stem
    paper_dir = output_dir / paper_id
    paper_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'#' * 70}")
    print(f"# PAPER: {paper_id}")
    print(f"{'#' * 70}")

    # ── Step 0: Load paper ──
    print("[Step 0] Loading PDF...")
    pdf_extractor = MistralPDFExtractor(structured=False)
    paper_text = load_file_text(pdf_path, pdf_extractor)
    si_text = ""
    si_path = find_si_file(pdf_path)
    if si_path:
        try:
            si_text = load_file_text(si_path, pdf_extractor)
            print(f"  Found SI: {si_path.name} ({len(si_text):,} chars)")
        except Exception as e:
            print(f"  [WARN] SI load failed: {e}")

    paper = Paper(
        name=paper_id, id=paper_id, publication_text=paper_text, si_text=si_text
    )
    print(f"  Text: {len(paper_text):,} chars")

    # ── Step 1: Extract materials ──
    print("[Step 1] Extracting materials...")
    material_sig = make_dspy_text_extractor_signature(
        instructions=(
            "Extract ALL distinct superconducting material compositions "
            "that were synthesized and tested in this paper. "
            "If the paper studies multiple variants "
            "(e.g., different doping levels x=0.1, x=0.2, x=0.3), "
            "list EACH variant as a separate material."
        ),
        output_name="materials",
        output_description=(
            "ALL distinct synthesized material compositions as a "
            "comma-separated list using chemical formulas. "
            "Include doping levels and stoichiometry."
        ),
    )
    material_lm = get_llm_from_name(
        "gemini-3.0-pro", model_kwargs={"temperature": 0.0, "max_tokens": 16000}
    )
    material_extractor = DspyTextExtractor(
        signature=material_sig, lm=material_lm
    )
    text_for_llm = clean_text(paper.publication_text)
    materials_text = material_extractor.forward(input=text_for_llm)
    materials = smart_split_materials(materials_text)
    print(
        f"  Found {len(materials)} materials: "
        f"{materials[:5]}{'...' if len(materials) > 5 else ''}"
    )

    # ── Step 2: Extract synthesis ──
    print("[Step 2] Extracting synthesis procedures...")
    from llm_synthesis.metrics.judge.general_synthesis_judge import (
        DspyGeneralSynthesisJudge,
        make_general_synthesis_judge_signature,
    )
    from llm_synthesis.models.paper import SynthesisEntry
    from llm_synthesis.transformers.synthesis_extraction import (
        dspy_synthesis_extraction as _dse_module,
    )
    dspy_synthesis_extractor_cls = _dse_module.DspySynthesisExtractor
    make_dspy_synthesis_extractor_signature = (
        _dse_module.make_dspy_synthesis_extractor_signature
    )

    synthesis_system_prompt = (
        "You are a helpful assistant that extracts structured "
        "synthesis procedures from scientific papers.\n\n"
        "IMPORTANT: For the synthesis_method field, you MUST choose "
        "from these exact values:\n"
        "'PVD', 'CVD', 'arc discharge', 'ball milling',\n"
        "'spray pyrolysis', 'electrospinning',\n"
        "'sol-gel', 'hydrothermal', 'solvothermal', "
        "'precipitation', 'coprecipitation', 'combustion',\n"
        "'microwave-assisted', 'sonochemical', "
        "'template-directed', 'solid-state', 'flux growth',\n"
        "'float zone & Bridgman', "
        "'arc melting & induction melting', "
        "'spark plasma sintering',\n"
        "'electrochemical deposition', "
        "'chemical bath deposition', "
        "'liquid-phase epitaxy', 'self-assembly',\n"
        "'atomic layer deposition', 'molecular beam epitaxy', "
        "'pulsed laser deposition', 'ion implantation',\n"
        "'lithographic patterning', 'wet impregnation', "
        "'incipient wetness impregnation', 'mechanical mixing',\n"
        "'solution-based', 'mechanochemical', 'other'\n\n"
        "For the target_compound_type field, you MUST choose from "
        "these exact values:\n"
        "'metals & alloys', 'ceramics & glasses', "
        "'polymers & soft matter', 'composites',\n"
        "'semiconductors & electronic', 'nanomaterials', "
        "'two-dimensional materials',\n"
        "'framework & porous materials', "
        "'biomaterials & biological', 'liquid materials',\n"
        "'hybrid & organic-inorganic', "
        "'functional materials & catalysts', "
        "'energy & sustainability',\n"
        "'smart & responsive materials', "
        "'emerging & quantum materials', 'other'\n\n"
        "If the exact method is not in the list, use the closest "
        "match or 'other'."
    )

    synthesis_sig = make_dspy_synthesis_extractor_signature(
        instructions=(
            "Extract the complete structured synthesis procedure for "
            "the specified material. Include all steps, conditions "
            "(temperature, time, atmosphere), equipment, and "
            "precursors. If a step is not explicitly mentioned in "
            "the text, do not hallucinate details."
        ),
    )

    synthesis_lm = get_llm_from_name(
        GEMINI_MODEL,
        model_kwargs={
            "temperature": 0.0,
            "max_tokens": 32000,
            "max_retries": 3,
        },
        system_prompt=synthesis_system_prompt,
    )
    synthesis_extractor = dspy_synthesis_extractor_cls(
        signature=synthesis_sig, lm=synthesis_lm
    )

    judge_lm = get_llm_from_name(
        GEMINI_MODEL,
        model_kwargs={"temperature": 0.0, "max_tokens": 16000},
    )
    judge_sig = make_general_synthesis_judge_signature()
    judge = DspyGeneralSynthesisJudge(signature=judge_sig, lm=judge_lm)

    all_syntheses = []
    for i, material in enumerate(materials, 1):
        print(f"  [{i}/{len(materials)}] {material}")
        try:
            synthesis = synthesis_extractor.forward(
                input=(text_for_llm, material)
            )
            try:
                evaluation = judge.forward(
                    (text_for_llm, json.dumps(synthesis.model_dump()), material)
                )
            except Exception as e:
                print(f"    [WARN] Judge failed: {e}")
                evaluation = None
            all_syntheses.append(
                SynthesisEntry(
                    material=material,
                    synthesis=synthesis,
                    evaluation=evaluation,
                )
            )
            print(
                f"    Method: {synthesis.synthesis_method}, "
                f"Steps: {len(synthesis.steps)}"
            )
        except Exception as e:
            print(f"    [ERROR] {e}")
            all_syntheses.append(
                SynthesisEntry(
                    material=material,
                    synthesis=None,
                    evaluation=None,
                )
            )
    n_synth = sum(1 for e in all_syntheses if e.synthesis is not None)
    print(f"  Extracted synthesis for {n_synth}/{len(materials)} materials")

    # ── Step 3: Extract Tc from text (multi-condition) ──
    print("[Step 3] Extracting Tc from text (multi-condition)...")

    # Build materials context for Tc extraction
    mat_list_str = ", ".join(
        materials[:30]
    )  # cap at 30 to avoid bloating prompt
    materials_context = (
        f"\n\n5. MATERIALS IN THIS PAPER (from prior extraction):\n"
        f"   {mat_list_str}\n"
        f"   You MUST report a Tc line for EACH of these materials.\n"
        f"   If a material has no Tc mentioned anywhere in the "
        f"   text, report it with\n"
        f"   superconducting: NO and all Tc fields as NR.\n"
        f"   If additional materials with Tc values appear in "
        f"the text but are not in\n"
        f"   this list, include them too."
    )

    tc_text_sig = make_dspy_text_extractor_signature(
        signature_name="TextToTcMulti",
        instructions=(
            "Extract ALL superconducting critical temperature (Tc) "
            "values from this paper.\n\n"
            "RULES:\n"
            "1. NO HALLUCINATION: Only extract Tc values that appear "
            "as explicit numbers in\n"
            "   this paper's own results. Never use general "
            "knowledge. Never extract values\n"
            "   cited from other references. If no number is stated,"
            " report NR.\n\n"
            "2. Tc NOTATIONS — look for all of these:\n"
            "   onset: Tc_onset, Tconset, Tc,onset, Tc(onset), "
            "T_c^onset\n"
            "   midpoint: Tc, Tc_mid, Tc,mid, T_c^mid\n"
            "   zero-resistance: Tc_zero, Tc,zero, Tc(0), T_c^zero\n"
            "   transition width: ΔTc, delta_Tc — report this as "
            "delta_Tc, NOT as a Tc value.\n\n"
            "3. WHERE TO FIND Tc — check ALL of these locations in "
            "the paper:\n"
            "   - Abstract (often states the main Tc result)\n"
            "   - Results / Discussion sections "
            "(detailed Tc values per sample)\n"
            "   - Tables (Tc columns, summary tables of properties)\n"
            "   - Figure captions (e.g., 'Tc = 39 K as shown in Fig. 3')\n"
            "   - Conclusions (often restates key Tc findings)\n"
            "   Also look for INDIRECT phrasings:\n"
            "   - 'superconductivity emerges below 39 K'\n"
            "   - 'becomes superconducting at 23 K'\n"
            "   - 'critical temperature of 92 K'\n"
            "   - 'superconducting transition at 7.2 K'\n"
            "   - 'zero resistance below 25 K'\n"
            "   - 'onset of superconductivity near 30 K'\n\n"
            "4. MATERIAL NAME = fully resolved chemical formula.\n"
            "   Substitute all variables with their numeric values.\n"
            "   e.g. A(B1-xCx) with x=0.3 → AB0.7C0.3\n"
            "   Each distinct composition gets its own row.\n"
            "   Use the chemical formula, not sample labels "
            "(S1, SC2, etc.).\n\n"
            "5. CONDITION = only external factors NOT encoded in the "
            "formula:\n"
            "   pressure, magnetic field, sample form "
            "(single-crystal, thin-film, etc.).\n"
            "   If none apply, use 'ambient'." + materials_context
        ),
        input_description=(
            "The full publication text from a superconductivity paper."
        ),
        output_name="tc_values",
        output_description=(
            "One line per (material, condition) pair in this "
            "pipe-delimited format:\n"
            "formula | condition: <value or ambient> "
            "| superconducting: YES/NO "
            "| T_onset: <number> K | Tc: <number> K "
            "| T_zero: <number> K | delta_Tc: <number> K\n\n"
            "Use NR for any value not explicitly reported in the "
            "text.\n\n"
            "Example lines:\n"
            "YBa2Cu3O7 | condition: ambient | superconducting: YES "
            "| T_onset: 93 K | Tc: 92 K | T_zero: 91 K "
            "| delta_Tc: NR\n"
            "Nb3Sn | condition: 12 GPa | superconducting: YES "
            "| T_onset: NR | Tc: 23 K | T_zero: NR | delta_Tc: NR\n"
            "SrTiO3 | condition: ambient | superconducting: NO "
            "| T_onset: NR | Tc: NR | T_zero: NR | delta_Tc: NR"
        ),
    )
    tc_text_lm = get_llm_from_name(
        GEMINI_MODEL, model_kwargs={"temperature": 0.0, "max_tokens": 16384}
    )
    tc_text_extractor = DspyTextExtractor(signature=tc_text_sig, lm=tc_text_lm)
    max_text_chars = 60_000
    tc_input_text = (
        text_for_llm[:max_text_chars]
        if len(text_for_llm) > max_text_chars
        else text_for_llm
    )
    try:
        tc_text_raw = tc_text_extractor.forward(input=tc_input_text)
    except Exception as e:
        print(f"  [WARN] Tc text extraction failed: {e}")
        tc_text_raw = ""
    tc_from_text = parse_tc_text_response_multi(tc_text_raw)
    n_text_tc = sum(1 for v in tc_from_text if v.get("Tc_mid") is not None)
    n_conditions = len(
        set((e["material"], e["condition"]) for e in tc_from_text)
    )
    print(
        f"  Found {n_text_tc} Tc values across "
        f"{n_conditions} (material, condition) pairs"
    )

    # ── Steps 3-7: Figures + Plot data + Filtering + VLM Tc ──
    figures = []
    plots = []
    plot_figures = []
    relevant_plots = []
    tc_from_vlm_original = {}
    tc_from_vlm_snippet = {}
    plot_mappings = []

    if not skip_figures:
        # Step 3: Extract figures
        print("[Step 4] Extracting figures (Florence-2)...")
        from llm_synthesis.transformers.figure_extraction import (
            FigureExtractorMarkdown,
        )

        extractor = FigureExtractorMarkdown(
            segmenter="florence",
            florence_repo_id="amayuelas/plot-visualization-florence-2-lora-32",
        )
        figures = extractor.forward(paper.publication_text)
        print(f"  Found {len(figures)} subfigures")

        if figures:
            # Step 4: Extract plot data
            print("[Step 5] Extracting plot data (Claude VLM)...")
            from llm_synthesis.models.figure import FigureInfoWithPaper
            from llm_synthesis.transformers.plot_extraction \
                .claude_extraction.plot_data_extraction import (
                ClaudeLinePlotDataExtractor,
            )
            from llm_synthesis.utils.figure_utils import clean_text_from_images

            plot_extractor = ClaudeLinePlotDataExtractor(
                model_name=CLAUDE_MODEL, max_tokens=8000
            )
            for i, fig in enumerate(figures):
                try:
                    fig_with_paper = FigureInfoWithPaper(
                        base64_data=fig.base64_data,
                        alt_text=fig.alt_text,
                        position=fig.position,
                        context_before=fig.context_before,
                        context_after=fig.context_after,
                        figure_reference=fig.figure_reference,
                        figure_class=fig.figure_class,
                        quantitative=fig.quantitative,
                        paper_text=clean_text_from_images(
                            paper.publication_text
                        ),
                        si_text=paper.si_text,
                    )
                    plot_data = plot_extractor.forward(fig_with_paper)
                    if plot_data and plot_data.name_to_coordinates:
                        plots.append(plot_data)
                        plot_figures.append(fig)
                except Exception as e:
                    print(f"    [ERROR] Figure {i}: {e}")
            print(
                f"  Extracted data from {len(plots)} plots "
                f"(cost: ${plot_extractor.get_cost():.4f})"
            )

            # Step 5: Filter R(T) plots
            print("[Step 6] Filtering for R(T) plots...")
            # Reload filter config in case source was updated
            import importlib

            import llm_synthesis.config.plot_filter_config as _pfc_mod

            importlib.reload(_pfc_mod)
            filter_config = _pfc_mod.PlotFilterConfig.for_superconductivity()
            plot_filter = PlotFilter(filter_config)

            # Debug: show axis metadata for each plot
            for pi, p in enumerate(plots):
                print(
                    f"    Plot {pi}: x='{p.x_axis_label}' [{p.x_axis_unit}]  "
                    f"y='{p.y_left_axis_label}' [{p.y_left_axis_unit}]  "
                    f"series={list(p.name_to_coordinates.keys())[:3]}"
                )

            relevant_plots, skip_counts = plot_filter.filter_plots(
                plots, log_skipped=True
            )
            if skip_counts:
                for reason, cnt in skip_counts.items():
                    if cnt > 0:
                        print(f"    Skipped {cnt} plots: {reason}")

            # Fallback for missing axis metadata
            rejected_indices = {i for i in range(len(plots))} - {
                idx for idx, _ in relevant_plots
            }
            for i in sorted(rejected_indices):
                plot = plots[i]
                fig = plot_figures[i]
                x_missing = _is_axis_missing(
                    plot.x_axis_label, plot.x_axis_unit
                )
                y_missing = _is_axis_missing(
                    plot.y_left_axis_label, plot.y_left_axis_unit
                )
                if (x_missing or y_missing) and fallback_check_rt_plot(
                    plot, fig
                ):
                    relevant_plots.append((i, plot))
                    print(f"    Plot {i}: recovered via fallback context check")
            relevant_plots.sort(key=lambda x: x[0])
            print(f"  R(T) plots: {len(relevant_plots)} / {len(plots)} total")

            # Step 6: VLM Tc extraction — DUAL (original + snippet)
            if relevant_plots:
                print(
                    "[Step 7] Extracting Tc from R(T) plots — "
                    "DUAL (original + snippet)..."
                )
                _anthropic_client = anthropic.Anthropic()
                _cost_original = 0.0
                _cost_snippet = 0.0

                def _img_type(b64: str) -> str:
                    return (
                        "image/jpeg" if b64.startswith("/9j/") else "image/png"
                    )

                def _call_vlm(
                    fig_b64: str, prompt: str, crop_b64: str | None = None
                ) -> tuple[str, float]:
                    """Claude VLM call — single image or two images
                    (full + crop).
                    """
                    content = []
                    if crop_b64:
                        content.append(
                            {
                                "type": "text",
                                "text": "Image 1 — FULL R(T) plot:",
                            }
                        )
                    content.append(
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": _img_type(fig_b64),
                                "data": fig_b64,
                            },
                        }
                    )
                    if crop_b64:
                        content.append(
                            {
                                "type": "text",
                                "text": (
                                    "Image 2 — ZOOMED bottom-left quadrant "
                                    "(low T, low R region):"
                                ),
                            }
                        )
                        content.append(
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": _img_type(crop_b64),
                                    "data": crop_b64,
                                },
                            }
                        )
                    content.append({"type": "text", "text": prompt})

                    msg = _anthropic_client.messages.create(
                        model=CLAUDE_MODEL,
                        max_tokens=4096,
                        temperature=0.0,
                        messages=[{"role": "user", "content": content}],
                    )
                    cost = (msg.usage.input_tokens / 1e6) * 3.0 + (
                        msg.usage.output_tokens / 1e6
                    ) * 15.0
                    return msg.content[0].text, cost

                for idx, plot in relevant_plots:
                    fig = plot_figures[idx]
                    known_series = list(plot.name_to_coordinates.keys())
                    ref = fig.figure_reference or f"Plot {idx}"
                    caption = extract_figure_caption(
                        ref, paper.publication_text
                    )

                    print(f"\n  Plot {idx}: {ref}  (series: {known_series})")

                    # ── A) ORIGINAL (single image) ──
                    prompt_orig = build_tc_prompt(
                        known_series_names=known_series,
                        use_snippet=False,
                        figure_caption=caption,
                        materials=materials,
                    )
                    try:
                        resp, cost = _call_vlm(fig.base64_data, prompt_orig)
                        _cost_original += cost
                        parsed = sanity_check_delta_tc(
                            parse_direct_tc_response(resp)
                        )
                        parsed = clean_vlm_tc_results(parsed)
                        tc_from_vlm_original[idx] = parsed
                        for sn, v in parsed.items():
                            sc = "Y" if v.get("superconducting") else "N"
                            tc = (
                                f"{v['tc_mid']:.1f} K"
                                if v.get("tc_mid")
                                else "—"
                            )
                            print(f"    [ORIG] {sn}: SC={sc} Tc={tc}")
                    except Exception as e:
                        print(f"    [ORIG ERROR] {e}")

                    # ── B) SNIPPET (full + crop) ──
                    crop_b64 = crop_bottom_left_quadrant(fig.base64_data)
                    prompt_snip = build_tc_prompt(
                        known_series_names=known_series,
                        use_snippet=True,
                        figure_caption=caption,
                        materials=materials,
                    )
                    try:
                        resp, cost = _call_vlm(
                            fig.base64_data, prompt_snip, crop_b64=crop_b64
                        )
                        _cost_snippet += cost
                        parsed = sanity_check_delta_tc(
                            parse_direct_tc_response(resp)
                        )
                        parsed = clean_vlm_tc_results(parsed)
                        tc_from_vlm_snippet[idx] = parsed
                        for sn, v in parsed.items():
                            sc = "Y" if v.get("superconducting") else "N"
                            tc = (
                                f"{v['tc_mid']:.1f} K"
                                if v.get("tc_mid")
                                else "—"
                            )
                            print(f"    [SNIP] {sn}: SC={sc} Tc={tc}")
                    except Exception as e:
                        print(f"    [SNIP ERROR] {e}")

                print(
                    f"\n  VLM cost: Original ${_cost_original:.4f}"
                    f" | Snippet ${_cost_snippet:.4f}"
                    f" | Total ${_cost_original + _cost_snippet:.4f}"
                )

            # Step 7: Link series to materials
            if relevant_plots:
                print("[Step 8] Linking series to materials...")
                from llm_synthesis.models.performance import (
                    PlotMaterialMapping,
                )
                from llm_synthesis.transformers.performance_linking \
                    .base import (
                    LinkingInput,
                )
                from llm_synthesis.transformers.performance_linking \
                    .series_material_linker import (
                    SeriesMaterialLinker,
                )

                linker_lm = get_llm_from_name(
                    LINKER_MODEL,
                    model_kwargs={"temperature": 0.0, "max_tokens": 32000},
                )
                series_linker = SeriesMaterialLinker(lm=linker_lm)
                for idx, plot in relevant_plots:
                    fig = plot_figures[idx]
                    series_names = list(plot.name_to_coordinates.keys())
                    context = f"{fig.context_before} {fig.context_after}"
                    plot_meta = {
                        "title": plot.title,
                        "x_axis_label": plot.x_axis_label,
                        "x_axis_unit": plot.x_axis_unit,
                        "y_left_axis_label": plot.y_left_axis_label,
                        "y_left_axis_unit": plot.y_left_axis_unit,
                    }
                    linking_input = LinkingInput(
                        materials=materials,
                        series_names=series_names,
                        context=context,
                        plot_metadata=plot_meta,
                    )
                    validated = series_linker.forward(linking_input)
                    matched = {m.series_name for m in validated}
                    unmatched = [s for s in series_names if s not in matched]
                    plot_mappings.append(
                        PlotMaterialMapping(
                            plot_index=idx,
                            figure_reference=fig.figure_reference,
                            mappings=validated,
                            unmatched_series=unmatched,
                        )
                    )
                    for m in validated:
                        print(f"    '{m.series_name}' -> '{m.material_name}'")
    else:
        print("[Steps 4-8] Skipped (--skip-figures)")

    # ── Step 8: Aggregate + save per-paper JSON ──
    print("[Step 9] Aggregating and saving results...")
    performance_data = {}
    if plot_mappings and plots:
        performance_data = aggregate_all_materials_performance(
            materials, plot_mappings, plots
        )

    # Build VLM Tc lookup — BOTH original and snippet
    vlm_tc_per_material_orig = {}
    vlm_tc_per_material_snip = {}
    for mapping in plot_mappings:
        plot_idx = mapping.plot_index
        for sm in mapping.mappings:
            # Original
            if plot_idx in tc_from_vlm_original:
                vlm_data = _find_vlm_tc_for_series(
                    sm.series_name,
                    tc_from_vlm_original[plot_idx],
                    material_name=sm.material_name,
                )
                if vlm_data:
                    existing = vlm_tc_per_material_orig.get(sm.material_name)
                    if existing is None or (
                        _vlm_data_has_tc(vlm_data)
                        and not _vlm_data_has_tc(existing)
                    ):
                        vlm_tc_per_material_orig[sm.material_name] = vlm_data
            # Snippet
            if plot_idx in tc_from_vlm_snippet:
                vlm_data = _find_vlm_tc_for_series(
                    sm.series_name,
                    tc_from_vlm_snippet[plot_idx],
                    material_name=sm.material_name,
                )
                if vlm_data:
                    existing = vlm_tc_per_material_snip.get(sm.material_name)
                    if existing is None or (
                        _vlm_data_has_tc(vlm_data)
                        and not _vlm_data_has_tc(existing)
                    ):
                        vlm_tc_per_material_snip[sm.material_name] = vlm_data

    # Collect unmatched VLM series
    # (have Tc but weren't linked to any Step 1 material)
    linked_series = set()
    for mapping in plot_mappings:
        for sm in mapping.mappings:
            linked_series.add((mapping.plot_index, sm.series_name))

    vlm_only_materials = []
    vlm_only_materials_set = set()  # track normalized names to avoid duplicates
    # Also track which materials already have VLM data from linking
    all_vlm_material_norms = set()
    for name in list(vlm_tc_per_material_orig.keys()) + list(
        vlm_tc_per_material_snip.keys()
    ):
        all_vlm_material_norms.add(_normalize_formula(name))

    for plot_idx in set(
        list(tc_from_vlm_original.keys()) + list(tc_from_vlm_snippet.keys())
    ):
        orig_results = tc_from_vlm_original.get(plot_idx, {})
        snip_results = tc_from_vlm_snippet.get(plot_idx, {})
        all_series_names = set(
            list(orig_results.keys()) + list(snip_results.keys())
        )
        for series_name in all_series_names:
            if (plot_idx, series_name) in linked_series:
                continue  # already linked
            # Check if this series has any Tc
            orig_data = orig_results.get(series_name, {})
            snip_data = snip_results.get(series_name, {})
            has_tc = _vlm_data_has_tc(orig_data) or _vlm_data_has_tc(snip_data)
            if not has_tc:
                continue  # no Tc, skip
            # Validate that the series name looks like a real material formula
            if not is_valid_material_name(series_name):
                continue  # skip garbage names like "5 T", "x = 0.1", etc.
            # Check if already covered by normalized name
            norm = _normalize_formula(series_name)
            if norm in all_vlm_material_norms or norm in vlm_only_materials_set:
                continue
            vlm_only_materials_set.add(norm)
            vlm_only_materials.append(series_name)
            # Store their VLM data
            if orig_data:
                vlm_tc_per_material_orig[series_name] = orig_data
            if snip_data:
                vlm_tc_per_material_snip[series_name] = snip_data

    if vlm_only_materials:
        print(
            f"  Found {len(vlm_only_materials)} additional materials "
            "from VLM Tc "
            f"(unlinked series): {vlm_only_materials[:5]}"
        )

    # Fuzzy-match text Tc (multi-condition)
    text_tc_per_material = {}
    for material in materials:
        matches = find_matching_text_tc_multi(material, tc_from_text)
        if matches:
            text_tc_per_material[material] = matches

    # Collect text-only materials not found by Step 1 material extraction
    matched_text_materials = set()
    for entries in text_tc_per_material.values():
        for e in entries:
            matched_text_materials.add(_normalize_formula(e["material"]))
    text_only_materials = []
    for entry in tc_from_text:
        if not is_valid_material_name(entry["material"]):
            continue  # skip garbage names from text extraction
        norm = _normalize_formula(entry["material"])
        if norm not in matched_text_materials:
            matched_text_materials.add(norm)
            text_only_materials.append(entry["material"])
    # Add text-only materials with their entries
    for mat in text_only_materials:
        matches = find_matching_text_tc_multi(mat, tc_from_text)
        if matches:
            text_tc_per_material[mat] = matches
    if text_only_materials:
        print(
            f"  Found {len(text_only_materials)} additional materials "
            "from text Tc "
            f"(not in Step 1): {text_only_materials[:5]}"
        )

    # Combined material list: Step 1 + VLM-only + text-only (deduplicated)
    all_material_norms_seen = set(_normalize_formula(m) for m in materials)
    deduped_vlm_only = []
    for m in vlm_only_materials:
        norm = _normalize_formula(m)
        if norm not in all_material_norms_seen:
            all_material_norms_seen.add(norm)
            deduped_vlm_only.append(m)
    deduped_text_only = []
    for m in text_only_materials:
        norm = _normalize_formula(m)
        if norm not in all_material_norms_seen:
            all_material_norms_seen.add(norm)
            deduped_text_only.append(m)
    all_materials = materials + deduped_vlm_only + deduped_text_only

    # ── Merge materials with duplicate normalized forms ──
    # Prefer Step 1 names as canonical, merge VLM/text data into them
    norm_to_canonical = {}
    merged_materials = []
    for m in all_materials:
        norm = _normalize_formula(m)
        if norm in norm_to_canonical:
            canonical = norm_to_canonical[norm]
            # Merge VLM data from duplicate into canonical name
            if (
                m in vlm_tc_per_material_orig
                and canonical not in vlm_tc_per_material_orig
            ):
                vlm_tc_per_material_orig[canonical] = vlm_tc_per_material_orig[
                    m
                ]
            if (
                m in vlm_tc_per_material_snip
                and canonical not in vlm_tc_per_material_snip
            ):
                vlm_tc_per_material_snip[canonical] = vlm_tc_per_material_snip[
                    m
                ]
            if m in text_tc_per_material:
                if canonical not in text_tc_per_material:
                    text_tc_per_material[canonical] = text_tc_per_material[m]
                else:
                    existing_keys = {
                        (_normalize_formula(e["material"]), e.get("condition"))
                        for e in text_tc_per_material[canonical]
                    }
                    for e in text_tc_per_material[m]:
                        key = (
                            _normalize_formula(e["material"]),
                            e.get("condition"),
                        )
                        if key not in existing_keys:
                            text_tc_per_material[canonical].append(e)
            print(f"  Merged duplicate: '{m}' -> '{canonical}'")
        else:
            norm_to_canonical[norm] = m
            merged_materials.append(m)
    if len(merged_materials) < len(all_materials):
        print(
            f"  Deduplicated materials: {len(all_materials)} -> "
            f"{len(merged_materials)}"
        )
    all_materials = merged_materials

    # Build synthesis lookup by material
    synth_by_material = {}
    for entry in all_syntheses:
        synth_by_material[entry.material] = entry

    # Save per-material JSON
    for material in all_materials:
        text_entries = text_tc_per_material.get(material, [])
        vlm_orig = vlm_tc_per_material_orig.get(material, {})
        vlm_snip = vlm_tc_per_material_snip.get(material, {})
        synth_entry = synth_by_material.get(material)
        result = {
            "material": material,
            "synthesis": synth_entry.synthesis.model_dump()
            if synth_entry and synth_entry.synthesis
            else None,
            "evaluation": synth_entry.evaluation.model_dump()
            if synth_entry and synth_entry.evaluation
            else None,
            "tc_from_text": text_entries if text_entries else None,
            "tc_from_vlm_original": {
                "superconducting": vlm_orig.get("superconducting"),
                "T_onset": vlm_orig.get("t_onset"),
                "Tc_mid": vlm_orig.get("tc_mid"),
                "T_zero": vlm_orig.get("t_zero"),
                "Delta_Tc": vlm_orig.get("delta_tc"),
            }
            if vlm_orig
            else None,
            "tc_from_vlm_snippet": {
                "superconducting": vlm_snip.get("superconducting"),
                "T_onset": vlm_snip.get("t_onset"),
                "Tc_mid": vlm_snip.get("tc_mid"),
                "T_zero": vlm_snip.get("t_zero"),
                "Delta_Tc": vlm_snip.get("delta_tc"),
            }
            if vlm_snip
            else None,
            "performance": performance_data[material].model_dump()
            if material in performance_data
            else None,
        }
        mat_name = sanitize_filename(material)
        with open(paper_dir / f"{mat_name}.json", "w") as f:
            json.dump(result, f, indent=2, default=str)

    # Save summary
    summary = {
        "paper_id": paper.id,
        "total_materials": len(all_materials),
        "materials_list": all_materials,
        "materials_from_step1": len(materials),
        "materials_from_text_only": len(text_only_materials),
        "materials_with_synthesis": sum(
            1 for e in all_syntheses if e.synthesis is not None
        ),
        "total_plots_extracted": len(plots),
        "rt_plots_found": len(relevant_plots),
        "materials_with_text_tc": sum(
            1
            for m in all_materials
            if any(
                e.get("Tc_mid") is not None
                for e in text_tc_per_material.get(m, [])
            )
        ),
        "materials_with_vlm_tc_orig": sum(
            1 for v in vlm_tc_per_material_orig.values() if _vlm_data_has_tc(v)
        ),
        "materials_with_vlm_tc_snip": sum(
            1 for v in vlm_tc_per_material_snip.values() if _vlm_data_has_tc(v)
        ),
        "total_text_conditions": sum(
            len(v) for v in text_tc_per_material.values()
        ),
    }
    with open(paper_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # ── Step 10: Build flat records ──
    year = extract_year_from_arxiv_id(paper.id)
    flat_records = []

    for material in all_materials:
        text_entries = text_tc_per_material.get(material, [])
        vlm_orig = vlm_tc_per_material_orig.get(material, {})
        vlm_snip = vlm_tc_per_material_snip.get(material, {})

        # Synthesis info
        synth_entry = synth_by_material.get(material)
        synth_method = (
            synth_entry.synthesis.synthesis_method
            if synth_entry and synth_entry.synthesis
            else None
        )
        synth_score = (
            synth_entry.evaluation.scores.overall_score
            if synth_entry
            and synth_entry.evaluation
            and synth_entry.evaluation.scores
            else None
        )
        has_synth = (
            synth_entry is not None and synth_entry.synthesis is not None
        )

        # VLM values (shared across conditions — VLM doesn't know
        # about conditions)
        vlm_orig_tc = vlm_orig.get("tc_mid")
        vlm_orig_onset = vlm_orig.get("t_onset")
        vlm_orig_zero = vlm_orig.get("t_zero")
        _ = vlm_orig.get("superconducting")  # unused; fetched for symmetry
        vlm_orig_source = (
            vlm_orig.get("source", "main plot") if vlm_orig else None
        )

        vlm_snip_tc = vlm_snip.get("tc_mid")
        vlm_snip_onset = vlm_snip.get("t_onset")
        vlm_snip_zero = vlm_snip.get("t_zero")
        _ = vlm_snip.get("superconducting")  # unused; fetched for symmetry
        vlm_snip_source = (
            vlm_snip.get("source", "main plot") if vlm_snip else None
        )

        vlm_source_plot = None
        for mapping in plot_mappings:
            for sm in mapping.mappings:
                if sm.material_name == material:
                    vlm_source_plot = mapping.figure_reference
                    break
            if vlm_source_plot:
                break

        # If we have text entries with conditions, create one row per condition
        if text_entries:
            for te in text_entries:
                text_tc = te.get("Tc_mid")
                text_onset = te.get("T_onset")
                text_zero = te.get("T_zero")
                _ = te.get("superconducting")  # unused; derived separately
                condition = te.get("condition", "ambient")

                # Derive is_superconductor from actual Tc values
                is_sc = derive_is_superconductor(
                    tc_text=text_tc,
                    tc_text_onset=text_onset,
                    tc_text_zero=text_zero,
                    vlm_orig_tc=vlm_orig_tc,
                    vlm_snip_tc=vlm_snip_tc,
                )

                tc_best, tc_best_source = pick_best_tc(
                    text_tc, vlm_orig_tc, vlm_snip_tc, text_onset
                )

                flat_records.append(
                    {
                        "paper_id": paper.id,
                        "year": year,
                        "material": material,
                        "material_normalized": normalize_formula_for_csv(
                            material
                        ),
                        "condition": condition,
                        "is_superconductor": is_sc,
                        "tc_text": text_tc,
                        "tc_text_onset": text_onset,
                        "tc_text_zero": text_zero,
                        "tc_text_source": None,
                        "tc_vlm_orig": vlm_orig_tc,
                        "tc_vlm_orig_onset": vlm_orig_onset,
                        "tc_vlm_orig_zero": vlm_orig_zero,
                        "tc_vlm_orig_source": vlm_orig_source,
                        "tc_vlm_snip": vlm_snip_tc,
                        "tc_vlm_snip_onset": vlm_snip_onset,
                        "tc_vlm_snip_zero": vlm_snip_zero,
                        "tc_vlm_snip_source": vlm_snip_source,
                        "tc_vlm_source_plot": vlm_source_plot,
                        "tc_best": tc_best,
                        "tc_best_source": tc_best_source,
                        "has_text_tc": text_tc is not None,
                        "has_vlm_tc_orig": vlm_orig_tc is not None,
                        "has_vlm_tc_snip": vlm_snip_tc is not None,
                        "has_synthesis": has_synth,
                        "synthesis_method": synth_method,
                        "synthesis_score": synth_score,
                    }
                )
        else:
            # No text Tc — still create a row with VLM data
            is_sc = derive_is_superconductor(
                vlm_orig_tc=vlm_orig_tc,
                vlm_snip_tc=vlm_snip_tc,
            )

            tc_best, tc_best_source = pick_best_tc(
                None, vlm_orig_tc, vlm_snip_tc
            )

            flat_records.append(
                {
                    "paper_id": paper.id,
                    "year": year,
                    "material": material,
                    "material_normalized": normalize_formula_for_csv(material),
                    "condition": "ambient",
                    "is_superconductor": is_sc,
                    "tc_text": None,
                    "tc_text_onset": None,
                    "tc_text_zero": None,
                    "tc_text_source": None,
                    "tc_vlm_orig": vlm_orig_tc,
                    "tc_vlm_orig_onset": vlm_orig_onset,
                    "tc_vlm_orig_zero": vlm_orig_zero,
                    "tc_vlm_orig_source": vlm_orig_source,
                    "tc_vlm_snip": vlm_snip_tc,
                    "tc_vlm_snip_onset": vlm_snip_onset,
                    "tc_vlm_snip_zero": vlm_snip_zero,
                    "tc_vlm_snip_source": vlm_snip_source,
                    "tc_vlm_source_plot": vlm_source_plot,
                    "tc_best": tc_best,
                    "tc_best_source": tc_best_source,
                    "has_text_tc": False,
                    "has_vlm_tc_orig": vlm_orig_tc is not None,
                    "has_vlm_tc_snip": vlm_snip_tc is not None,
                    "has_synthesis": has_synth,
                    "synthesis_method": synth_method,
                    "synthesis_score": synth_score,
                }
            )

    # ── Deduplicate flat_records by (material_normalized, condition) ──
    seen_records = {}  # key -> index into deduped list
    deduped_records = []
    for rec in flat_records:
        key = (rec["material_normalized"], rec.get("condition", "ambient"))
        if key in seen_records:
            existing_idx = seen_records[key]
            existing = deduped_records[existing_idx]

            # Keep the record with the most non-None Tc fields
            def _tc_count(r):
                return sum(
                    1
                    for f in [
                        "tc_text",
                        "tc_text_onset",
                        "tc_text_zero",
                        "tc_vlm_orig",
                        "tc_vlm_snip",
                        "tc_best",
                    ]
                    if r.get(f) is not None
                )

            if _tc_count(rec) > _tc_count(existing):
                deduped_records[existing_idx] = rec
        else:
            seen_records[key] = len(deduped_records)
            deduped_records.append(rec)
    if len(deduped_records) < len(flat_records):
        print(
            f"  Deduplicated records: {len(flat_records)} -> "
            f"{len(deduped_records)}"
        )
    flat_records = deduped_records

    # ── Drop rows with no Tc data at all ──
    _tc_fields = [
        "tc_text",
        "tc_text_onset",
        "tc_text_zero",
        "tc_vlm_orig",
        "tc_vlm_orig_onset",
        "tc_vlm_orig_zero",
        "tc_vlm_snip",
        "tc_vlm_snip_onset",
        "tc_vlm_snip_zero",
    ]
    before_drop = len(flat_records)
    flat_records = [
        r for r in flat_records if any(r.get(f) is not None for f in _tc_fields)
    ]
    if len(flat_records) < before_drop:
        print(
            f"  Dropped {before_drop - len(flat_records)} rows with no Tc data"
        )

    # Save JSONL
    with open(paper_dir / "tc_flat_records.jsonl", "w") as f:
        for rec in flat_records:
            f.write(json.dumps(rec, default=str, indent=2) + "\n")

    # Print summary table
    print(
        f"\n  {'Material':<30} {'Cond.':<15} {'SC?':<5} "
        f"{'Tc_txt':>7} {'Tc_orig':>8} {'Tc_snip':>8} "
        f"{'Tc_best':>8} {'Source':<12} {'Synth?':<7} "
        f"{'Method':<15}"
    )
    print(f"  {'-' * 130}")
    for rec in flat_records:
        sc = (
            "YES"
            if rec["is_superconductor"]
            else ("NO" if rec["is_superconductor"] is False else "?")
        )
        tc_t = f"{rec['tc_text']:.1f}" if rec["tc_text"] else "—"
        tc_o = f"{rec['tc_vlm_orig']:.1f}" if rec["tc_vlm_orig"] else "—"
        tc_s = f"{rec['tc_vlm_snip']:.1f}" if rec["tc_vlm_snip"] else "—"
        tc_b = f"{rec['tc_best']:.1f}" if rec["tc_best"] else "—"
        cond = rec["condition"][:14]
        has_s = "YES" if rec.get("has_synthesis") else "NO"
        s_meth = (rec.get("synthesis_method") or "—")[:14]
        print(
            f"  {rec['material']:<30} {cond:<15} {sc:<5} "
            f"{tc_t:>7} {tc_o:>8} {tc_s:>8} {tc_b:>8} "
            f"{rec['tc_best_source']:<12} {has_s:<7} {s_meth:<15}"
        )

    return flat_records


# =============================================================================
# MASTER CSV MANAGEMENT
# =============================================================================


def append_to_master_csv(flat_records: list[dict], master_path: Path):
    """Append records to master CSV, replacing existing rows
    for the same paper.
    """
    master_path.parent.mkdir(parents=True, exist_ok=True)

    existing_keys = set()
    if master_path.exists() and master_path.stat().st_size > 0:
        with open(master_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_keys.add(
                    (
                        row.get("paper_id", ""),
                        row.get("material", ""),
                        row.get("condition", ""),
                    )
                )

    new_keys = {
        (r["paper_id"], r["material"], r.get("condition", ""))
        for r in flat_records
    }
    replace_keys = existing_keys & new_keys

    if replace_keys:
        all_rows = []
        if master_path.exists():
            with open(master_path, newline="") as f:
                reader = csv.DictReader(f)
                all_rows = [
                    row
                    for row in reader
                    if (
                        row.get("paper_id", ""),
                        row.get("material", ""),
                        row.get("condition", ""),
                    )
                    not in replace_keys
                ]
        all_rows.extend(
            {k: (str(v) if v is not None else "") for k, v in r.items()}
            for r in flat_records
        )
        with open(master_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=CSV_COLUMNS, quoting=csv.QUOTE_ALL
            )
            writer.writeheader()
            writer.writerows(all_rows)
    else:
        write_header = (
            not master_path.exists() or master_path.stat().st_size == 0
        )
        with open(master_path, "a", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=CSV_COLUMNS, quoting=csv.QUOTE_ALL
            )
            if write_header:
                writer.writeheader()
            for rec in flat_records:
                writer.writerow(
                    {
                        k: (str(v) if v is not None else "")
                        for k, v in rec.items()
                    }
                )


# =============================================================================
# MAIN
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Batch Tc extraction (snippet-enhanced + synthesis).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "folder", type=str, help="Path to folder containing PDF papers"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output directory (default: <folder>/results_snippet)",
    )
    parser.add_argument(
        "--max",
        type=int,
        default=None,
        help="Max number of papers to process (randomly sampled)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip papers that already have results",
    )
    parser.add_argument(
        "--skip-figures",
        action="store_true",
        help="Skip figure extraction (text-only, no VLM)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible sampling with --max",
    )
    args = parser.parse_args()

    folder = Path(args.folder).resolve()
    if not folder.is_dir():
        print(f"ERROR: {folder} is not a directory")
        sys.exit(1)

    if args.output:
        output_dir = Path(args.output).resolve()
    else:
        output_dir = folder / "results_snippet"
    output_dir.mkdir(parents=True, exist_ok=True)
    master_csv = output_dir / "tc_master_snippet.csv"

    # Discover PDFs
    pdfs = sorted(folder.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {folder}")
        sys.exit(1)

    # Skip existing if requested
    if args.skip_existing:
        already_done = {
            d.name
            for d in output_dir.iterdir()
            if d.is_dir() and (d / "summary.json").exists()
        }
        pdfs = [p for p in pdfs if p.stem not in already_done]
        if not pdfs:
            print(
                "All papers already processed. Use without "
                "--skip-existing to re-run."
            )
            sys.exit(0)

    # Random sample if --max is given
    if args.max and args.max < len(pdfs):
        rng = random.Random(args.seed)
        pdfs = rng.sample(pdfs, args.max)
        pdfs.sort(key=lambda p: p.name)  # sort for readable output

    print(f"{'=' * 70}")
    print("BATCH Tc EXTRACTION (snippet-enhanced + synthesis)")
    print(f"{'=' * 70}")
    print(f"  Folder:       {folder}")
    print(f"  Output:       {output_dir}")
    print(f"  Master CSV:   {master_csv}")
    print(f"  PDFs to run:  {len(pdfs)}")
    print(f"  Skip figures: {args.skip_figures}")
    if args.seed is not None:
        print(f"  Random seed:  {args.seed}")
    print(f"{'=' * 70}")

    # ── Dependency check ──
    errors = []
    try:
        from transformers import CLIPImageProcessor  # noqa: F401
    except (ImportError, ModuleNotFoundError):
        if not args.skip_figures:
            errors.append(
                "transformers.CLIPImageProcessor not found. "
                "Fix: pip install --upgrade transformers"
            )
    try:
        import anthropic  # noqa: F401
    except ImportError:
        errors.append("anthropic SDK not installed. Fix: pip install anthropic")
    try:
        import dspy  # noqa: F401
    except ImportError:
        errors.append("dspy not installed. Fix: pip install dspy")
    if errors:
        for e in errors:
            print(f"  [ERROR] {e}")
        sys.exit(1)
    print("[OK] Dependencies verified\n")

    # ── Process each paper ──
    all_flat_records = []
    results_log = []
    t_total_start = time.time()

    for i, pdf_path in enumerate(pdfs, 1):
        print(f"\n{'=' * 70}")
        print(f"  [{i}/{len(pdfs)}] {pdf_path.name}")
        print(f"{'=' * 70}")

        t_start = time.time()
        try:
            flat_records = process_one_paper(
                pdf_path, output_dir, skip_figures=args.skip_figures
            )
            elapsed = time.time() - t_start

            # Append to master CSV
            append_to_master_csv(flat_records, master_csv)
            all_flat_records.extend(flat_records)

            n_tc = sum(1 for r in flat_records if r["tc_best"] is not None)
            results_log.append(
                {
                    "paper": pdf_path.stem,
                    "status": "OK",
                    "records": len(flat_records),
                    "with_tc": n_tc,
                    "time_s": f"{elapsed:.0f}",
                }
            )
            print(
                f"\n  [OK] {pdf_path.stem}: {len(flat_records)} records, "
                f"{n_tc} with Tc ({elapsed:.0f}s)"
            )

        except Exception as e:
            elapsed = time.time() - t_start
            results_log.append(
                {
                    "paper": pdf_path.stem,
                    "status": f"FAILED: {e}",
                    "records": 0,
                    "with_tc": 0,
                    "time_s": f"{elapsed:.0f}",
                }
            )
            print(f"\n  [FAILED] {pdf_path.stem}: {e}")
            traceback.print_exc()

    # ── Final summary ──
    t_total = time.time() - t_total_start
    print(f"\n\n{'=' * 70}")
    print(f"BATCH COMPLETE — {len(pdfs)} papers in {t_total:.0f}s")
    print(f"{'=' * 70}")
    print(f"{'Paper':<45} {'Status':<10} {'Recs':>5} {'Tc':>5} {'Time':>6}")
    print(f"{'-' * 70}")
    for r in results_log:
        status = r["status"][:8]
        print(
            f"{r['paper']:<45} {status:<10} "
            f"{r['records']:>5} {r['with_tc']:>5} {r['time_s']:>5}s"
        )

    total_recs = sum(r["records"] for r in results_log)
    total_tc = sum(r["with_tc"] for r in results_log)
    n_ok = sum(1 for r in results_log if r["status"] == "OK")
    n_fail = len(results_log) - n_ok
    print(f"{'-' * 70}")
    print(f"  Total: {n_ok} succeeded, {n_fail} failed")
    print(f"  Records: {total_recs} total, {total_tc} with Tc")
    print(f"  Master CSV: {master_csv} ({total_recs} rows)")


if __name__ == "__main__":
    main()
