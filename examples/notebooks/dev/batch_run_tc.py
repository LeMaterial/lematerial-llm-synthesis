#!/usr/bin/env python3
"""
Batch Tc Extraction Script
===========================
Runs the full superconductivity Tc extraction pipeline on every PDF in a folder.

Usage:
    python batch_run_tc.py /path/to/pdf_folder
    python batch_run_tc.py /path/to/pdf_folder --max 5          # first 5 only
    python batch_run_tc.py /path/to/pdf_folder --skip-existing  # skip already-processed
    python batch_run_tc.py /path/to/pdf_folder --skip-figures   # text-only (no VLM)

Results are saved to <pdf_folder>/results/<paper_id>/ and appended to
<pdf_folder>/results/tc_master.csv.
"""

import argparse
import csv
import json
import os
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

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env", override=True)

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
import logging
logging.getLogger("pydantic").setLevel(logging.ERROR)
logging.getLogger("LiteLLM").setLevel(logging.ERROR)
logging.getLogger("litellm").setLevel(logging.ERROR)

# ── Model config ──
GEMINI_MODEL = "gemini-3.0-flash"
CLAUDE_MODEL = "claude-sonnet-4-20250514"
LINKER_MODEL = "gemini-3.0-flash"


# =============================================================================
# HELPERS (extracted from notebook)
# =============================================================================

SI_PATTERNS = ["_SI", "-SI", "_si", "-si", "_Supporting", "_supporting",
               "_Supplementary", "_supplementary", "_supp", "_Supp"]


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
            from llm_synthesis.transformers.pdf_extraction import MistralPDFExtractor
            pdf_extractor = MistralPDFExtractor(structured=False)
        with open(path, "rb") as f:
            return pdf_extractor.forward(f.read())
    elif suffix in [".md", ".txt"]:
        with open(path, "r", errors="replace") as f:
            return f.read()
    else:
        raise ValueError(f"Unsupported file type: {suffix}")


def fallback_check_rt_plot(plot, fig) -> bool:
    context = f"{fig.context_before or ''} {fig.context_after or ''} {fig.alt_text or ''}".lower()
    rt_context_hints = [
        "resistivity", "resistance", "ρ(t)", "ρ vs", "r(t)", "r vs t",
        "t (k)", "t [k]", "temperature dependence of ρ",
        "temperature dependence of the resistivity",
        "temperature dependence of the resistance",
        "μω cm", "μω·cm", "mω cm",
    ]
    return any(hint in context for hint in rt_context_hints)


def _is_axis_missing(label, unit) -> bool:
    return (not label or not label.strip()) and (not unit or not unit.strip())


# ── Tc text parsing ──

def parse_tc_text_response(raw_text: str) -> dict:
    results = {}
    for line in raw_text.strip().split("\n"):
        line = line.strip()
        if not line or "|" not in line:
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 2:
            continue
        material = parts[0].strip()
        entry = {"superconducting": False, "T_onset": None, "Tc_mid": None, "T_zero": None}
        for part in parts[1:]:
            part_lower = part.lower().strip()
            if "superconducting" in part_lower:
                entry["superconducting"] = "yes" in part_lower
            else:
                match = re.match(r"(t_onset|tc_mid|tc|t_zero)\s*:\s*(\d+\.?\d*)\s*k?", part_lower)
                if match:
                    key = match.group(1)
                    val = float(match.group(2))
                    if key in ("tc", "tc_mid"):
                        entry["Tc_mid"] = val
                    elif key == "t_onset":
                        entry["T_onset"] = val
                    elif key == "t_zero":
                        entry["T_zero"] = val
        results[material] = entry
    return results


# ── VLM Tc prompt + parsing ──

DIRECT_TC_PROMPT_TEMPLATE = """
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
  - Resistance drops from a finite value all the way to ZERO (or very close to zero)
  - It looks like a cliff or step function, not a gradual slope

If you see a curve that gradually decreases over 10-50 K, that is NOT
superconductivity — it is normal metallic/Kondo behavior.

STEP 0 — EXAMINE THE FULL FIGURE (main plot + any insets/panels):
a) Identify ALL panels in the figure: the main plot and any insets, secondary
   panels, or embedded sub-plots. For each one, describe:
     - What quantity is on each axis (e.g., ρ vs T, Tc vs x, phase diagram, etc.)
     - The axis ranges and tick marks
     - Whether it contains information relevant to determining Tc

b) CATEGORIZE each inset/panel into one of these types:
     (i)   ZOOMED R(T): A magnified view of the transition region of the same
           R(T) data as the main plot. Has the same axes (R vs T) but a narrower
           temperature range. → Use this PREFERENTIALLY for geometric Tc
           construction (much better spatial resolution).
     (ii)  Tc SUMMARY: Shows Tc (or T_onset, T_zero) as a function of composition,
           pressure, doping, field, etc. (e.g., "Tc vs x" or a phase diagram with
           a Tc boundary). → Read Tc values DIRECTLY from this panel for each
           series/composition. These are typically the most accurate values available.
     (iii) OTHER: Any panel not directly useful for Tc (e.g., Hall coefficient,
           magnetization, dR/dT, crystal structure). → Note it but do not use it
           for Tc determination.

c) Read ALL numbered tick marks on the main plot axes. If a zoomed inset exists,
   read its tick marks separately.

d) CRITICAL — SCALE AWARENESS: Before reading ANY value from the main plot,
   explicitly note the x-axis range. If the temperature axis spans a wide range
   (e.g., 0–300 K) and the transitions happen in a small fraction of that range,
   be extra careful with interpolation. Prefer reading from a zoomed inset or
   Tc-summary inset if available — the main plot resolution may be too poor to
   determine precise transition temperatures.

STEP 0.5 — EXTRACT Tc FROM SUMMARY INSETS (if any type-(ii) inset found):
If you identified a Tc-summary inset in Step 0b(ii), read the Tc values directly
from it for each series/composition. Report them as:
  inset_tc_<series_name>: <value> K

These values serve as a REFERENCE. In Step 4, if your geometric construction
from the R(T) curve gives a Tc that differs from the inset value by more than
20%, trust the inset value and report it instead — noting "source: inset".

STEP 1 — IDENTIFY SERIES:
{series_name_instruction}
List every distinct curve visible in the plot.

STEP 2 — READ RESISTANCE VALUES AT LOWEST AND HIGHEST TEMPERATURE:
For EACH series, read two values CAREFULLY:
  a) R_at_lowest_T: the resistance/resistivity value at the LOWEST
     temperature shown in the plot. Look at the leftmost data point of
     this series — what y-value does it have? Read the y-axis scale
     carefully. If the leftmost point is at y=0 on the scale, that is zero.
     If it is at y=5, y=10, y=20, etc. — that is NOT zero.
  b) R_at_highest_T: the resistance/resistivity at the HIGHEST temperature
     (rightmost data point, i.e., normal-state value).

Report both values for every series. This is critical for determining
which materials are superconducting.

STEP 3 — CONFIRM SUPERCONDUCTIVITY:
A series is superconducting ONLY if R_at_lowest_T is approximately zero
(within measurement noise of the zero line on the y-axis).

IMPORTANT: Be strict about what "approximately zero" means:
  - If the y-axis goes from 0 to 30 μΩ cm, then R_at_lowest_T must be
    below ~0.5 μΩ cm to count as zero (i.e., visually touching the x-axis)
  - A value like 5, 10, or 20 μΩ cm is NOT approximately zero
  - A curve that decreases gradually but never reaches the bottom of the
    plot is NOT superconducting

For non-superconducting series, report:
  superconducting: NO
  reason: <e.g., "R_at_lowest_T = 5 μΩ cm, clearly above zero">

STEP 4 — GEOMETRIC Tc CONSTRUCTION (only for confirmed superconductors):
Use the INSET if available (better resolution), otherwise use the main plot.

For each superconducting series:

  a) NORMAL-STATE LEVEL: Read R_normal from the plateau IMMEDIATELY above
     the sharp superconducting drop. This is NOT the maximum resistance at
     high temperature — it is the resistance just before the sharp drop begins.

  b) FIND T_onset: The temperature where the SHARP superconducting drop
     begins. This is where resistance SUDDENLY starts falling toward zero
     (not where it gradually decreases due to metallic behavior).

  c) FIND T_zero: Moving right from low T along R ≈ 0, T_zero is the LAST
     point still at R ≈ 0 before resistance starts rising. T_zero < T_onset.

  d) SANITY CHECK: T_onset - T_zero should typically be 0.1 to 5 K for
     most superconductors. If your T_onset - T_zero > 10 K, you are
     probably measuring a gradual metallic decline, NOT the SC transition.
     Re-examine the plot.

  e) Tc_mid = (T_onset + T_zero) / 2

  f) Delta_Tc = T_onset - T_zero

  g) CROSS-CHECK WITH INSET: If you extracted inset Tc values in Step 0.5,
     compare your geometric Tc_mid with the inset value. If they differ by
     more than 20%, the inset value is more reliable — use it instead and
     set source to "inset".

STEP 5 — RELATIVE ORDERING OF TRANSITIONS:
Even when transitions appear close together on the plot, they are almost
never at EXACTLY the same temperature. Look carefully:

  a) For each PAIR of superconducting series, compare which one starts
     dropping FIRST (at higher T). Look at the actual data points/markers —
     which series still has high resistance when the other has already
     started dropping?

  b) If series A starts dropping at a higher T than series B, then
     T_onset(A) > T_onset(B). Even if the difference is small (0.1-1 K),
     report it — do NOT round both to the same value.

  c) NEVER report identical T_onset values for different series unless you
     are absolutely certain they are the same after careful comparison.

Output format:

inset_detected: <yes/no>
inset_type: <"zoomed_rt" | "tc_summary" | "other" | "none">
inset_description: <brief description of what the inset shows, or "N/A">
inset_axes: <tick marks of inset if present, otherwise "N/A">
inset_tc_values: <series: value K, series: value K, ... (if tc_summary type), otherwise "N/A">
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

relative_ordering: <which series transitions first (highest T_onset),
second, etc. — and by approximately how many K do they differ?>

Do not output any other text.
"""


def build_tc_prompt(known_series_names: list[str] | None = None) -> str:
    if known_series_names and len(known_series_names) > 0:
        names_list = "\n".join(f"  {i+1}. {name}" for i, name in enumerate(known_series_names))
        instruction = (
            "The following series were previously identified in this plot. "
            "You MUST use these EXACT names in your output (in the 'Series:' lines):\n"
            f"{names_list}\n\n"
            "If you see additional curves not in this list, add them with descriptive names."
        )
    else:
        instruction = "List every distinct curve (by legend label, color, marker)."
    return DIRECT_TC_PROMPT_TEMPLATE.format(series_name_instruction=instruction)


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
                    if np_lower == ks_lower or np_lower in ks_lower or ks_lower in np_lower:
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
                    f"VLM said SC=NO, but inset Tc={inset_tc:.1f} K found. Using inset value."
                )
            continue
        delta = vals.get("delta_tc")
        if delta is not None and delta > 10:
            inset_tc = vals.get("inset_tc")
            if inset_tc is not None and inset_tc > 0:
                corrected[series_name]["tc_mid"] = inset_tc
                corrected[series_name]["source"] = "inset"
                corrected[series_name]["_sanity_override"] = (
                    f"Delta_Tc={delta:.1f} K too wide (>10 K). Using inset Tc={inset_tc:.1f} K instead."
                )
                for k in ("t_onset", "t_zero", "delta_tc"):
                    corrected[series_name].pop(k, None)
            else:
                corrected[series_name]["superconducting"] = False
                corrected[series_name]["_sanity_override"] = (
                    f"Delta_Tc={delta:.1f} K is too wide (>10 K). "
                    f"Likely confusing metallic decline with SC transition."
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
                    f"Geometric Tc_mid={tc_mid:.1f} K differs from inset Tc={inset_tc:.1f} K "
                    f"by {relative_diff*100:.0f}% (>20%). Using inset value."
                )
    return corrected


# ── Fuzzy matching helpers ──

def _normalize_formula(s: str) -> str:
    base = re.sub(r'\s*\([^)]*\)\s*$', '', s).strip()
    base = base.replace('δ', 'delta').replace('Δ', 'delta')
    base = base.replace('₀', '0').replace('₁', '1').replace('₂', '2').replace('₃', '3')
    base = base.replace('₄', '4').replace('₅', '5').replace('₆', '6').replace('₇', '7')
    base = base.replace('₈', '8').replace('₉', '9').replace('₋', '-')
    return base.lower().replace(' ', '').replace('−', '-')


def _find_matching_text_tc(material: str, tc_from_text: dict) -> list:
    mat_norm = _normalize_formula(material)
    matches = []
    for tc_key, tc_entry in tc_from_text.items():
        key_norm = _normalize_formula(tc_key)
        if key_norm == mat_norm:
            matches.append((tc_key, tc_entry))
    matches.sort(key=lambda x: (not x[1].get("superconducting", False), -(x[1].get("Tc_mid") or 0)))
    return matches


def _find_vlm_tc_for_series(series_name: str, vlm_results: dict) -> dict:
    if series_name in vlm_results:
        return vlm_results[series_name]
    def _norm(s):
        s = s.replace('₂', '2').replace('₄', '4').replace('₇', '7').replace('₁', '1')
        s = s.replace('₃', '3').replace('₅', '5').replace('₆', '6').replace('₈', '8')
        s = s.replace('₉', '9').replace('₀', '0').replace('₋', '-').replace('δ', 'delta')
        return s.lower().replace(' ', '')
    sn = _norm(series_name)
    for vlm_key, vlm_val in vlm_results.items():
        vk = _norm(vlm_key)
        if sn in vk or vk in sn:
            return vlm_val
        sn_parts = re.split(r'[\s_-]+', series_name.lower())
        vk_parts = re.split(r'[\s_-]+', vlm_key.lower())
        if len(sn_parts) >= 1 and len(vk_parts) >= 1 and sn_parts[0] == vk_parts[0]:
            return vlm_val
    if len(vlm_results) == 1:
        return next(iter(vlm_results.values()))
    return {}


def _vlm_data_has_tc(vlm_data: dict) -> bool:
    return bool(vlm_data.get("superconducting") and vlm_data.get("tc_mid") is not None)


# ── CSV helpers ──

def extract_year_from_arxiv_id(paper_id: str) -> int | None:
    clean = paper_id.split("_")[0]
    clean = re.sub(r'v\d+$', '', clean)
    match = re.match(r'^(\d{2})(\d{2})\.\d+$', clean)
    if match:
        yy = int(match.group(1))
        return 2000 + yy if yy < 90 else 1900 + yy
    return None


def normalize_formula_for_csv(s: str) -> str:
    base = re.sub(r'\s*\([^)]*\)\s*$', '', s).strip()
    sub_map = {'₀': '0', '₁': '1', '₂': '2', '₃': '3', '₄': '4',
               '₅': '5', '₆': '6', '₇': '7', '₈': '8', '₉': '9',
               '₋': '-', '₊': '+', '₍': '(', '₎': ')'}
    for uni, asc in sub_map.items():
        base = base.replace(uni, asc)
    base = base.replace('δ', 'delta').replace('Δ', 'Delta')
    base = base.replace('−', '-').replace('–', '-')
    return base.strip()


def pick_best_tc(text_tc, vlm_tc, text_onset):
    if text_tc is not None:
        return text_tc, "text"
    if text_onset is not None:
        return text_onset, "text_onset"
    if vlm_tc is not None:
        return vlm_tc, "vlm"
    return None, "none"


CSV_COLUMNS = [
    "paper_id", "year", "material", "material_normalized",
    "is_superconductor",
    "tc_text", "tc_text_onset", "tc_text_zero", "tc_text_source",
    "tc_vlm", "tc_vlm_onset", "tc_vlm_zero", "tc_vlm_source",
    "tc_vlm_source_plot",
    "tc_best", "tc_best_source",
    "has_text_tc", "has_vlm_tc",
    "synthesis_method", "synthesis_score",
]


# =============================================================================
# PIPELINE — one paper
# =============================================================================

def process_one_paper(pdf_path: Path, output_dir: Path, skip_figures: bool = False):
    """Run the full Tc extraction pipeline on a single PDF. Returns flat records."""
    from llm_synthesis.models.paper import Paper, SynthesisEntry
    from llm_synthesis.transformers.pdf_extraction import MistralPDFExtractor
    from llm_synthesis.transformers.material_extraction.dspy_extraction import (
        DspyTextExtractor, make_dspy_text_extractor_signature,
    )
    from llm_synthesis.utils.dspy_utils import get_llm_from_name
    from llm_synthesis.utils.markdown_utils import clean_text
    from llm_synthesis.transformers.synthesis_extraction.dspy_synthesis_extraction import (
        DspySynthesisExtractor, make_dspy_synthesis_extractor_signature,
    )
    from llm_synthesis.metrics.judge.general_synthesis_judge import (
        DspyGeneralSynthesisJudge, make_general_synthesis_judge_signature,
    )
    from llm_synthesis.config.plot_filter_config import PlotFilterConfig
    from llm_synthesis.transformers.performance_linking.plot_filter import PlotFilter
    from llm_synthesis.services.llm_api.claude import ClaudeAPIClient
    from llm_synthesis.utils.performance_utils import (
        aggregate_all_materials_performance, sanitize_filename,
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

    paper = Paper(name=paper_id, id=paper_id,
                  publication_text=paper_text, si_text=si_text)
    print(f"  Text: {len(paper_text):,} chars")

    # ── Step 1: Extract materials ──
    print("[Step 1] Extracting materials...")
    material_sig = make_dspy_text_extractor_signature(
        instructions=(
            "Extract ALL distinct superconducting material compositions that were synthesized "
            "and tested in this paper. If the paper studies multiple variants "
            "(e.g., different doping levels x=0.1, x=0.2, x=0.3), list EACH variant "
            "as a separate material."
        ),
        output_name="materials",
        output_description=(
            "ALL distinct synthesized material compositions as a comma-separated list "
            "using chemical formulas. Include doping levels and stoichiometry."
        ),
    )
    material_lm = get_llm_from_name("gemini-3.0-pro",
                                     model_kwargs={"temperature": 0.0, "max_tokens": 16000})
    material_extractor = DspyTextExtractor(signature=material_sig, lm=material_lm)
    text_for_llm = clean_text(paper.publication_text)
    materials_text = material_extractor.forward(input=text_for_llm)
    materials = [m.strip() for m in materials_text.replace("\n", ",").split(",") if m.strip()]
    print(f"  Found {len(materials)} materials: {materials[:5]}{'...' if len(materials) > 5 else ''}")

    # ── Step 2: Extract synthesis ──
    print("[Step 2] Extracting synthesis...")
    SYNTHESIS_SYSTEM_PROMPT = (
        "You are a helpful assistant that extracts structured synthesis procedures from scientific papers.\n\n"
        "IMPORTANT: For the synthesis_method field, you MUST choose from these exact values:\n"
        "'PVD', 'CVD', 'arc discharge', 'ball milling', 'spray pyrolysis', 'electrospinning',\n"
        "'sol-gel', 'hydrothermal', 'solvothermal', 'precipitation', 'coprecipitation', 'combustion',\n"
        "'microwave-assisted', 'sonochemical', 'template-directed', 'solid-state', 'flux growth',\n"
        "'float zone & Bridgman', 'arc melting & induction melting', 'spark plasma sintering',\n"
        "'electrochemical deposition', 'chemical bath deposition', 'liquid-phase epitaxy', 'self-assembly',\n"
        "'atomic layer deposition', 'molecular beam epitaxy', 'pulsed laser deposition', 'ion implantation',\n"
        "'lithographic patterning', 'wet impregnation', 'incipient wetness impregnation', 'mechanical mixing',\n"
        "'solution-based', 'mechanochemical', 'other'\n\n"
        "For the target_compound_type field, you MUST choose from these exact values:\n"
        "'metals & alloys', 'ceramics & glasses', 'polymers & soft matter', 'composites',\n"
        "'semiconductors & electronic', 'nanomaterials', 'two-dimensional materials',\n"
        "'framework & porous materials', 'biomaterials & biological', 'liquid materials',\n"
        "'hybrid & organic-inorganic', 'functional materials & catalysts', 'energy & sustainability',\n"
        "'smart & responsive materials', 'emerging & quantum materials', 'other'"
    )
    synthesis_sig = make_dspy_synthesis_extractor_signature(
        instructions="Extract the complete structured synthesis procedure for the specified material.",
    )
    synthesis_lm = get_llm_from_name(GEMINI_MODEL,
                                      model_kwargs={"temperature": 0.0, "max_tokens": 32000, "max_retries": 3},
                                      system_prompt=SYNTHESIS_SYSTEM_PROMPT)
    synthesis_extractor = DspySynthesisExtractor(signature=synthesis_sig, lm=synthesis_lm)
    judge_lm = get_llm_from_name(GEMINI_MODEL, model_kwargs={"temperature": 0.1, "max_tokens": 16000})
    judge_sig = make_general_synthesis_judge_signature()
    judge = DspyGeneralSynthesisJudge(signature=judge_sig, lm=judge_lm)

    all_syntheses = []
    for i, material in enumerate(materials, 1):
        print(f"  [{i}/{len(materials)}] {material}...", end=" ", flush=True)
        try:
            synthesis = synthesis_extractor.forward(input=(text_for_llm, material))
            try:
                evaluation = judge.forward((text_for_llm, json.dumps(synthesis.model_dump()), material))
                print(f"OK (score={evaluation.scores.overall_score:.1f})")
            except Exception:
                evaluation = None
                print("OK (judge failed)")
            all_syntheses.append(SynthesisEntry(material=material, synthesis=synthesis, evaluation=evaluation))
        except Exception as e:
            print(f"ERROR: {e}")
            all_syntheses.append(SynthesisEntry(material=material, synthesis=None, evaluation=None))

    # ── Step 3: Extract Tc from text ──
    print("[Step 3] Extracting Tc from text...")
    tc_text_sig = make_dspy_text_extractor_signature(
        signature_name="TextToTc",
        instructions=(
            "Extract ALL critical temperature (Tc) values reported in this superconductivity paper. "
            "For EACH material that has a Tc value mentioned in the text, report:\n"
            "  - The material formula\n"
            "  - T_onset (if explicitly reported)\n"
            "  - Tc: the critical temperature\n"
            "  - T_zero (if explicitly reported)\n"
            "  - Whether the material is superconducting (YES/NO)\n\n"
            "Only extract values explicitly stated. Use 'NR' for values not reported."
        ),
        input_description="The full publication text from a superconductivity paper.",
        output_name="tc_values",
        output_description=(
            "For each material, one line in the format:\n"
            "material_formula | superconducting: YES/NO | T_onset: <value> K | Tc: <value> K | T_zero: <value> K\n"
            "Use 'NR' for values not reported."
        ),
    )
    tc_text_lm = get_llm_from_name(GEMINI_MODEL, model_kwargs={"temperature": 0.0, "max_tokens": 16384})
    tc_text_extractor = DspyTextExtractor(signature=tc_text_sig, lm=tc_text_lm)
    MAX_TEXT_CHARS = 60_000
    tc_input_text = text_for_llm[:MAX_TEXT_CHARS] if len(text_for_llm) > MAX_TEXT_CHARS else text_for_llm
    try:
        tc_text_raw = tc_text_extractor.forward(input=tc_input_text)
    except Exception as e:
        print(f"  [WARN] Tc text extraction failed: {e}")
        tc_text_raw = ""
    tc_from_text = parse_tc_text_response(tc_text_raw)
    n_text_tc = sum(1 for v in tc_from_text.values() if v.get("Tc_mid") is not None)
    print(f"  Found Tc for {n_text_tc}/{len(tc_from_text)} materials from text")

    # ── Steps 4-7: Figures + Plot data + Filtering + VLM Tc ──
    figures = []
    plots = []
    plot_figures = []
    relevant_plots = []
    tc_from_vlm = {}
    plot_mappings = []

    if not skip_figures:
        # Step 4: Extract figures
        print("[Step 4] Extracting figures (Florence-2)...")
        from llm_synthesis.transformers.figure_extraction import FigureExtractorMarkdown
        extractor = FigureExtractorMarkdown(
            segmenter="florence",
            florence_repo_id="amayuelas/plot-visualization-florence-2-lora-32",
        )
        figures = extractor.forward(paper.publication_text)
        print(f"  Found {len(figures)} subfigures")

        if figures:
            # Step 5: Extract plot data
            print("[Step 5] Extracting plot data (Claude VLM)...")
            from llm_synthesis.transformers.plot_extraction.claude_extraction.plot_data_extraction import (
                ClaudeLinePlotDataExtractor,
            )
            from llm_synthesis.models.figure import FigureInfoWithPaper
            from llm_synthesis.utils.figure_utils import clean_text_from_images

            plot_extractor = ClaudeLinePlotDataExtractor(model_name=CLAUDE_MODEL, max_tokens=8000)
            for i, fig in enumerate(figures):
                try:
                    fig_with_paper = FigureInfoWithPaper(
                        base64_data=fig.base64_data, alt_text=fig.alt_text,
                        position=fig.position, context_before=fig.context_before,
                        context_after=fig.context_after, figure_reference=fig.figure_reference,
                        figure_class=fig.figure_class, quantitative=fig.quantitative,
                        paper_text=clean_text_from_images(paper.publication_text),
                        si_text=paper.si_text,
                    )
                    plot_data = plot_extractor.forward(fig_with_paper)
                    if plot_data and plot_data.name_to_coordinates:
                        plots.append(plot_data)
                        plot_figures.append(fig)
                except Exception as e:
                    print(f"    [ERROR] Figure {i}: {e}")
            print(f"  Extracted data from {len(plots)} plots (cost: ${plot_extractor.get_cost():.4f})")

            # Step 6: Filter R(T) plots
            print("[Step 6] Filtering for R(T) plots...")
            filter_config = PlotFilterConfig.for_superconductivity()
            plot_filter = PlotFilter(filter_config)
            relevant_plots, skip_counts = plot_filter.filter_plots(plots, log_skipped=False)

            # Fallback for missing axis metadata
            rejected_indices = {i for i in range(len(plots))} - {idx for idx, _ in relevant_plots}
            for i in sorted(rejected_indices):
                plot = plots[i]
                fig = plot_figures[i]
                x_missing = _is_axis_missing(plot.x_axis_label, plot.x_axis_unit)
                y_missing = _is_axis_missing(plot.y_left_axis_label, plot.y_left_axis_unit)
                if (x_missing or y_missing) and fallback_check_rt_plot(plot, fig):
                    relevant_plots.append((i, plot))
            relevant_plots.sort(key=lambda x: x[0])
            print(f"  R(T) plots: {len(relevant_plots)} / {len(plots)} total")

            # Step 7: VLM Tc extraction
            if relevant_plots:
                print("[Step 7] Extracting Tc from R(T) plots (Claude VLM)...")
                claude_client = ClaudeAPIClient(CLAUDE_MODEL)
                for idx, plot in relevant_plots:
                    fig = plot_figures[idx]
                    known_series = list(plot.name_to_coordinates.keys())
                    prompt = build_tc_prompt(known_series_names=known_series)
                    try:
                        response = claude_client.vision_model_api_call(
                            figure_base64=fig.base64_data, prompt=prompt,
                            max_tokens=8000, temperature=0.0,
                        )
                        parsed = parse_direct_tc_response(response)
                        corrected = sanity_check_delta_tc(parsed)
                        tc_from_vlm[idx] = corrected
                        for sn, vals in corrected.items():
                            sc = "YES" if vals.get("superconducting") else "NO"
                            tc = vals.get("tc_mid")
                            tc_str = f"{tc:.1f} K" if tc else "N/A"
                            print(f"    {sn}: SC={sc}, Tc_mid={tc_str}")
                    except Exception as e:
                        print(f"    [ERROR] Plot {idx}: {e}")
                print(f"  VLM Tc cost: ${claude_client.get_cost():.4f}")

            # Step 8: Link series to materials
            if relevant_plots:
                print("[Step 8] Linking series to materials...")
                import dspy
                from llm_synthesis.transformers.performance_linking.series_material_linker import (
                    SeriesMaterialLinker,
                )
                from llm_synthesis.transformers.performance_linking.base import LinkingInput
                from llm_synthesis.models.performance import PlotMaterialMapping

                linker_lm = get_llm_from_name(LINKER_MODEL,
                                               model_kwargs={"temperature": 0.0, "max_tokens": 16000})
                series_linker = SeriesMaterialLinker(lm=linker_lm)
                for idx, plot in relevant_plots:
                    fig = plot_figures[idx]
                    series_names = list(plot.name_to_coordinates.keys())
                    context = f"{fig.context_before} {fig.context_after}"
                    plot_meta = {
                        "title": plot.title, "x_axis_label": plot.x_axis_label,
                        "x_axis_unit": plot.x_axis_unit,
                        "y_left_axis_label": plot.y_left_axis_label,
                        "y_left_axis_unit": plot.y_left_axis_unit,
                    }
                    linking_input = LinkingInput(materials=materials, series_names=series_names,
                                                 context=context, plot_metadata=plot_meta)
                    validated = series_linker.forward(linking_input)
                    matched = {m.series_name for m in validated}
                    unmatched = [s for s in series_names if s not in matched]
                    plot_mappings.append(PlotMaterialMapping(
                        plot_index=idx, figure_reference=fig.figure_reference,
                        mappings=validated, unmatched_series=unmatched,
                    ))
                    for m in validated:
                        print(f"    '{m.series_name}' → '{m.material_name}'")
    else:
        print("[Steps 4-8] Skipped (--skip-figures)")

    # ── Step 9: Aggregate ──
    print("[Step 9] Aggregating results...")
    performance_data = {}
    if plot_mappings and plots:
        performance_data = aggregate_all_materials_performance(materials, plot_mappings, plots)

    # Build VLM Tc lookup
    vlm_tc_per_material = {}
    for mapping in plot_mappings:
        plot_idx = mapping.plot_index
        if plot_idx not in tc_from_vlm:
            continue
        vlm_results_for_plot = tc_from_vlm[plot_idx]
        for sm in mapping.mappings:
            vlm_data = _find_vlm_tc_for_series(sm.series_name, vlm_results_for_plot)
            if vlm_data:
                existing = vlm_tc_per_material.get(sm.material_name)
                if existing is None:
                    vlm_tc_per_material[sm.material_name] = vlm_data
                elif _vlm_data_has_tc(vlm_data) and not _vlm_data_has_tc(existing):
                    vlm_tc_per_material[sm.material_name] = vlm_data

    # Fuzzy-match text Tc
    text_tc_per_material = {}
    for material in materials:
        matches = _find_matching_text_tc(material, tc_from_text)
        if matches:
            best_key, best_entry = matches[0]
            text_tc_per_material[material] = best_entry
            text_tc_per_material[material]["_all_variants"] = [
                {"condition": k, **v} for k, v in matches
            ]

    # ── Step 10: Save per-paper results ──
    print("[Step 10] Saving results...")
    for entry in all_syntheses:
        mat = entry.material
        text_tc_entry = text_tc_per_material.get(mat, {})
        text_tc_clean = {k: v for k, v in text_tc_entry.items() if not k.startswith("_")}
        vlm_tc_entry = vlm_tc_per_material.get(mat, {})
        result = {
            "material": mat,
            "synthesis": entry.synthesis.model_dump() if entry.synthesis else None,
            "evaluation": entry.evaluation.model_dump() if entry.evaluation else None,
            "tc_from_text": text_tc_clean if text_tc_clean else None,
            "tc_from_vlm": {
                "superconducting": vlm_tc_entry.get("superconducting"),
                "T_onset": vlm_tc_entry.get("t_onset"),
                "Tc_mid": vlm_tc_entry.get("tc_mid"),
                "T_zero": vlm_tc_entry.get("t_zero"),
                "Delta_Tc": vlm_tc_entry.get("delta_tc"),
            } if vlm_tc_entry else None,
            "performance": performance_data[mat].model_dump() if mat in performance_data else None,
        }
        mat_name = sanitize_filename(mat)
        with open(paper_dir / f"{mat_name}.json", "w") as f:
            json.dump(result, f, indent=2, default=str)

    # Save summary
    summary = {
        "paper_id": paper.id, "total_materials": len(materials),
        "materials_list": materials,
        "total_plots_extracted": len(plots),
        "rt_plots_found": len(relevant_plots),
        "materials_with_text_tc": sum(1 for v in text_tc_per_material.values() if v.get("Tc_mid") is not None),
        "materials_with_vlm_tc": sum(1 for v in vlm_tc_per_material.values() if _vlm_data_has_tc(v)),
    }
    with open(paper_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # ── Step 11: Build flat records ──
    year = extract_year_from_arxiv_id(paper.id)
    flat_records = []
    for entry in all_syntheses:
        mat = entry.material
        text_entry = text_tc_per_material.get(mat, {})
        vlm_entry = vlm_tc_per_material.get(mat, {})
        text_tc = text_entry.get("Tc_mid")
        text_onset = text_entry.get("T_onset")
        text_zero = text_entry.get("T_zero")
        text_sc = text_entry.get("superconducting")
        vlm_tc = vlm_entry.get("tc_mid")
        vlm_onset = vlm_entry.get("t_onset")
        vlm_zero = vlm_entry.get("t_zero")
        vlm_sc = vlm_entry.get("superconducting")
        vlm_source = vlm_entry.get("source", "main plot") if vlm_entry else None

        vlm_source_plot = None
        for mapping in plot_mappings:
            for sm in mapping.mappings:
                if sm.material_name == mat:
                    vlm_source_plot = mapping.figure_reference
                    break
            if vlm_source_plot:
                break

        if text_sc is not None:
            is_sc = text_sc
        elif vlm_sc is not None:
            is_sc = vlm_sc
        else:
            is_sc = None

        tc_best, tc_best_source = pick_best_tc(text_tc, vlm_tc, text_onset)
        synth_method = entry.synthesis.synthesis_method if entry.synthesis else None
        synth_score = (entry.evaluation.scores.overall_score
                       if entry.evaluation and entry.evaluation.scores else None)

        flat_records.append({
            "paper_id": paper.id, "year": year, "material": mat,
            "material_normalized": normalize_formula_for_csv(mat),
            "is_superconductor": is_sc,
            "tc_text": text_tc, "tc_text_onset": text_onset, "tc_text_zero": text_zero,
            "tc_text_source": None,
            "tc_vlm": vlm_tc, "tc_vlm_onset": vlm_onset, "tc_vlm_zero": vlm_zero,
            "tc_vlm_source": vlm_source, "tc_vlm_source_plot": vlm_source_plot,
            "tc_best": tc_best, "tc_best_source": tc_best_source,
            "has_text_tc": text_tc is not None, "has_vlm_tc": vlm_tc is not None,
            "synthesis_method": synth_method, "synthesis_score": synth_score,
        })

    # Save JSONL
    with open(paper_dir / "tc_flat_records.jsonl", "w") as f:
        for rec in flat_records:
            f.write(json.dumps(rec, default=str, indent=2) + "\n")

    # Print summary table
    print(f"\n  {'Material':<35} {'SC?':<5} {'Tc_text':>8} {'Tc_VLM':>8} {'Tc_best':>8} {'Source':<12}")
    print(f"  {'-'*85}")
    for rec in flat_records:
        sc = "YES" if rec["is_superconductor"] else ("NO" if rec["is_superconductor"] is False else "?")
        tc_t = f"{rec['tc_text']:.1f}" if rec["tc_text"] else "—"
        tc_v = f"{rec['tc_vlm']:.1f}" if rec["tc_vlm"] else "—"
        tc_b = f"{rec['tc_best']:.1f}" if rec["tc_best"] else "—"
        print(f"  {rec['material']:<35} {sc:<5} {tc_t:>8} {tc_v:>8} {tc_b:>8} {rec['tc_best_source']:<12}")

    return flat_records


# =============================================================================
# MASTER CSV MANAGEMENT
# =============================================================================

def append_to_master_csv(flat_records: list[dict], master_path: Path):
    """Append records to master CSV, replacing existing rows for the same paper."""
    master_path.parent.mkdir(parents=True, exist_ok=True)

    existing_keys = set()
    if master_path.exists() and master_path.stat().st_size > 0:
        with open(master_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_keys.add((row.get("paper_id", ""), row.get("material", "")))

    new_keys = {(r["paper_id"], r["material"]) for r in flat_records}
    replace_keys = existing_keys & new_keys

    if replace_keys:
        all_rows = []
        if master_path.exists():
            with open(master_path, "r", newline="") as f:
                reader = csv.DictReader(f)
                all_rows = [row for row in reader
                            if (row.get("paper_id", ""), row.get("material", "")) not in replace_keys]
        all_rows.extend({k: (str(v) if v is not None else "") for k, v in r.items()}
                        for r in flat_records)
        with open(master_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            writer.writerows(all_rows)
    else:
        write_header = not master_path.exists() or master_path.stat().st_size == 0
        with open(master_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            if write_header:
                writer.writeheader()
            for rec in flat_records:
                writer.writerow({k: (str(v) if v is not None else "") for k, v in rec.items()})


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Batch Tc extraction: run the full pipeline on all PDFs in a folder.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("folder", type=str,
                        help="Path to folder containing PDF papers")
    parser.add_argument("--max", type=int, default=None,
                        help="Max number of papers to process (default: all)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip papers that already have results")
    parser.add_argument("--skip-figures", action="store_true",
                        help="Skip figure extraction (text-only, no VLM)")
    args = parser.parse_args()

    folder = Path(args.folder).resolve()
    if not folder.is_dir():
        print(f"ERROR: {folder} is not a directory")
        sys.exit(1)

    output_dir = folder / "results"
    output_dir.mkdir(exist_ok=True)
    master_csv = output_dir / "tc_master.csv"

    # Discover PDFs
    pdfs = sorted(folder.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {folder}")
        sys.exit(1)

    # Skip existing if requested
    if args.skip_existing:
        already_done = {d.name for d in output_dir.iterdir() if d.is_dir() and (d / "summary.json").exists()}
        pdfs = [p for p in pdfs if p.stem not in already_done]
        if not pdfs:
            print("All papers already processed. Use without --skip-existing to re-run.")
            sys.exit(0)

    # Limit
    if args.max:
        pdfs = pdfs[:args.max]

    print(f"{'=' * 70}")
    print(f"BATCH Tc EXTRACTION")
    print(f"{'=' * 70}")
    print(f"  Folder:       {folder}")
    print(f"  Output:       {output_dir}")
    print(f"  Master CSV:   {master_csv}")
    print(f"  PDFs to run:  {len(pdfs)}")
    print(f"  Skip figures: {args.skip_figures}")
    print(f"{'=' * 70}")

    # ── Dependency check ──
    errors = []
    try:
        from transformers import CLIPImageProcessor  # noqa: F401
    except (ImportError, ModuleNotFoundError):
        if not args.skip_figures:
            errors.append("transformers.CLIPImageProcessor not found. Fix: pip install --upgrade transformers")
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
            flat_records = process_one_paper(pdf_path, output_dir, skip_figures=args.skip_figures)
            elapsed = time.time() - t_start

            # Append to master CSV
            append_to_master_csv(flat_records, master_csv)
            all_flat_records.extend(flat_records)

            n_tc = sum(1 for r in flat_records if r["tc_best"] is not None)
            results_log.append({
                "paper": pdf_path.stem, "status": "OK",
                "materials": len(flat_records), "with_tc": n_tc,
                "time_s": f"{elapsed:.0f}",
            })
            print(f"\n  [OK] {pdf_path.stem}: {len(flat_records)} materials, "
                  f"{n_tc} with Tc ({elapsed:.0f}s)")

        except Exception as e:
            elapsed = time.time() - t_start
            results_log.append({
                "paper": pdf_path.stem, "status": f"FAILED: {e}",
                "materials": 0, "with_tc": 0, "time_s": f"{elapsed:.0f}",
            })
            print(f"\n  [FAILED] {pdf_path.stem}: {e}")
            traceback.print_exc()

    # ── Final summary ──
    t_total = time.time() - t_total_start
    print(f"\n\n{'=' * 70}")
    print(f"BATCH COMPLETE — {len(pdfs)} papers in {t_total:.0f}s")
    print(f"{'=' * 70}")
    print(f"{'Paper':<45} {'Status':<10} {'Mats':>5} {'Tc':>5} {'Time':>6}")
    print(f"{'-' * 70}")
    for r in results_log:
        status = r["status"][:8]
        print(f"{r['paper']:<45} {status:<10} {r['materials']:>5} {r['with_tc']:>5} {r['time_s']:>5}s")

    total_mats = sum(r["materials"] for r in results_log)
    total_tc = sum(r["with_tc"] for r in results_log)
    n_ok = sum(1 for r in results_log if r["status"] == "OK")
    n_fail = len(results_log) - n_ok
    print(f"{'-' * 70}")
    print(f"  Total: {n_ok} succeeded, {n_fail} failed")
    print(f"  Materials: {total_mats} total, {total_tc} with Tc")
    print(f"  Master CSV: {master_csv} ({total_mats} rows)")


if __name__ == "__main__":
    main()
