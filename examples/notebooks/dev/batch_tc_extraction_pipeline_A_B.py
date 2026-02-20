"""
Batch Tc extraction from R(T) plots.

Runs Approach A (direct VLM) and Approach B (extract + compute) on all
try*.png images on the Desktop and outputs results to a CSV file.
"""

import base64
import csv
import os
import re
import sys
import traceback
import unicodedata

import numpy as np
from scipy.interpolate import UnivariateSpline

# Add project root to path and load .env
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, ".env"), override=True)

if not os.environ.get("ANTHROPIC_API_KEY"):
    print("ERROR: ANTHROPIC_API_KEY not set. Check .env file at:", os.path.join(PROJECT_ROOT, ".env"))
    sys.exit(1)

from llm_synthesis.services.llm_api.claude import ClaudeAPIClient
from llm_synthesis.transformers.plot_extraction.claude_extraction.plot_data_extraction import (
    ClaudeLinePlotDataExtractor,
)


# ─── Prompts ───────────────────────────────────────────────────────────────

DIRECT_TC_PROMPT = """
You are analyzing a Resistance (or Resistivity) vs Temperature plot from a
superconductivity paper. Your task is to determine the critical temperature
Tc for each series using the standard geometric construction.

STEP 0 — EXAMINE THE FIGURE:
a) Check if the figure contains an INSET or ZOOMED-IN panel that magnifies
   the transition region. Insets typically show a smaller temperature range
   (e.g., 30-40 K instead of 0-300 K) and are embedded within the main plot.
b) If an inset exists, use it PREFERENTIALLY for Tc determination — it has
   much better spatial resolution in the transition region.
c) Read ALL numbered tick marks on BOTH the main plot and the inset (if present).
   List them separately.

STEP 1 — IDENTIFY SERIES:
List every distinct curve (by legend label, color, marker).

STEP 2 — READ RESISTANCE VALUES AT LOWEST AND HIGHEST TEMPERATURE:
For EACH series, read two values:
  a) R_at_lowest_T: the resistance/resistivity value at the LOWEST
     temperature shown in the plot. Look at the leftmost data point of
     this series — what y-value does it have?
  b) R_at_highest_T: the resistance/resistivity at the HIGHEST temperature
     (rightmost data point, i.e., normal-state value).

Report both values for every series. This is critical for determining
which materials are superconducting.

STEP 3 — CONFIRM SUPERCONDUCTIVITY:
A series is superconducting ONLY if R_at_lowest_T is approximately zero.
  - If R_at_lowest_T is approximately 0 -> SUPERCONDUCTING
  - If R_at_lowest_T is clearly above zero (e.g., 0.1, 0.3, 1.0) ->
    NOT superconducting, even if the curve has sharp drops or kinks.

For non-superconducting series, report:
  superconducting: NO
  reason: <e.g., "R_at_lowest_T = 0.3, clearly above zero">

STEP 4 — GEOMETRIC Tc CONSTRUCTION (only for confirmed superconductors):
Use the INSET if available (better resolution), otherwise use the main plot.

For each superconducting series:

  a) NORMAL-STATE LEVEL: Read R_normal from the plateau above the transition.

  b) FIND T_onset: Moving left from the high-T plateau, T_onset is where
     resistance FIRST drops below R_normal. This is the HIGHEST temperature
     in the transition.

  c) FIND T_zero: Moving right from low T along R = 0, T_zero is the LAST
     point still at R = 0 before resistance starts rising. T_zero < T_onset.

  d) Tc_mid = (T_onset + T_zero) / 2

  e) Delta_Tc = T_onset - T_zero

STEP 5 — RELATIVE ORDERING OF TRANSITIONS:
Even when transitions appear close together on the plot, they are almost
never at EXACTLY the same temperature. Look carefully:

  a) For each PAIR of superconducting series, compare which one starts
     dropping FIRST (at higher T). Look at the actual data points/markers —
     which series still has high resistance when the other has already
     started dropping?

  b) If series A starts dropping at a higher T than series B, then
     T_onset(A) > T_onset(B). Even if the difference is small (1-3 K),
     report it — do NOT round both to the same value.

  c) Similarly for T_zero: which series reaches zero resistance at a higher T?

  d) NEVER report identical T_onset values for different series unless you
     are absolutely certain they are the same after careful comparison.

Output format:

inset_detected: <yes/no>
inset_axes: <tick marks of inset if present, otherwise "N/A">
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
source: <"inset" or "main plot">

relative_ordering: <which series transitions first (highest T_onset),
second, etc. — and by approximately how many K do they differ?>

Do not output any other text.
"""

EXTRACT_RT_PROMPT = """
You are extracting data points from a Resistance (or Resistivity) vs
Temperature plot. Extract the (T, R) coordinates of EVERY visible data
point for each series.

STEP 0 — EXAMINE THE FIGURE:
a) Check for an INSET or ZOOMED-IN panel. If present, note it but extract
   from the MAIN plot (which shows the full temperature range).
b) Read ALL numbered tick marks on the x-axis and y-axis.

STEP 1 — AXIS CALIBRATION (CRITICAL):
1. List every numbered tick mark on both axes.
2. For each data point, identify the TWO nearest tick marks on each axis
   and interpolate the position between them.

STEP 2 — IDENTIFY SERIES:
List every distinct curve (by legend label, color, marker style).

STEP 3 — EXTRACT DATA POINTS:
For EACH series, extract ALL visible data points as (T, R) pairs.

CRITICAL — SUPERCONDUCTING TRANSITIONS:
Some series may show a superconducting transition where resistance drops
sharply from a normal-state value to ZERO over a very narrow temperature
range. In the transition region:
  - Data points may be stacked nearly VERTICALLY (very similar T values
    but very different R values)
  - There may be 3-8 data points clustered at nearly the same T
  - Extract EACH of these points individually — they are the most
    important points in the entire plot
  - Do NOT skip or merge these closely-spaced points
  - Do NOT smooth over the transition by spreading points across a wider
    T range than they actually occupy

For each series, extract points from LOW T to HIGH T, including:
  - All points where R ~ 0 (superconducting state)
  - All points in the transition (the near-vertical cluster)
  - All points in the normal state above the transition
  - Points at the highest temperatures shown

Output format (one line per series):

Series_Name_1: [[T1, R1], [T2, R2], ...]
Series_Name_2: [[T1, R1], [T2, R2], ...]
title:
x_axis_label:
x_axis_unit:
y_left_axis_label:
y_left_axis_unit:

Do not output any other text, just the data in the format above.
"""


# ─── Helper functions ──────────────────────────────────────────────────────


def load_image_base64(img_path: str) -> str:
    """Load an image file and return base64-encoded string."""
    with open(img_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def parse_direct_tc_response(response_text: str) -> dict:
    """Parse the direct Tc prompt response."""
    results = {}
    current = None
    for line in response_text.strip().split("\n"):
        line = line.strip()
        if line.startswith("Series:"):
            current = line.split(":", 1)[1].strip()
            results[current] = {}
        elif current and ":" in line:
            key, val = line.split(":", 1)
            key = key.strip().lower().replace(" ", "_")
            val = val.strip()
            if key == "superconducting":
                results[current][key] = val.upper().startswith("YES")
            elif key in ("reason",):
                results[current][key] = val
            else:
                # Match a proper number (not just a lone ".")
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
    return results


_COLORS = {"red", "blue", "green", "black", "magenta", "cyan", "orange", "purple", "yellow"}
_MARKERS = {"squares", "circles", "triangles", "diamonds", "stars", "pentagons",
            "up-triangles", "down-triangles"}


def normalize_series_name(name: str) -> str:
    """
    Normalize a series name for fuzzy matching between Approach A and B.

    Strips unicode subscripts, ALL parenthetical annotations, color/marker words,
    leading letter prefixes, and normalizes separators so that e.g.:
      "b - Hf₈₀Fe₂₀"  and  "b_Hf80Fe20"
      "Ba0.5K0.5OFe2As2 (down-triangles, magenta)"  and  "Ba0.5K0.5OFe2As2"
      "Ca0.5Na0.5Fe2As2 (blue)"  and  "Blue_Ca0.5Na0.5Fe2As2"
    produce the same key.
    """
    s = name.strip()

    # Replace unicode subscript digits (₀₁₂₃₄₅₆₇₈₉) with normal digits
    subscript_map = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
    s = s.translate(subscript_map)

    # Remove parenthetical content that is ONLY colors/markers/combinations thereof
    # Keep parenthetical content with numbers/chemical info (e.g., "(0 T)", "(0.5 T)")
    color_marker_word = (
        r"(?:squares|circles|triangles|up-triangles|down-triangles|diamonds|stars|"
        r"red|blue|green|black|magenta|cyan|orange|purple|yellow)"
    )
    # Match parens containing only color/marker words separated by commas or spaces
    s = re.sub(
        r"\(\s*" + color_marker_word + r"(?:[\s,]+" + color_marker_word + r")*\s*\)",
        "", s, flags=re.IGNORECASE
    )

    # Strip remaining parentheses but KEEP their content (e.g., "(0 T)" → "0 T")
    # This handles field/condition annotations that are part of the identity
    s = s.replace("(", "").replace(")", "")

    # Remove leading single-letter prefix like "a - ", "b_", "c - "
    s = re.sub(r"^[a-g]\s*[-_]\s*", "", s)

    # Normalize separators: replace " - ", "_", multiple spaces → nothing
    s = re.sub(r"[\s_-]+", "", s)

    # Lowercase for comparison
    s = s.lower()

    # Remove color and marker words (handles "Blue_Ca..." → "ca..." and "ρab_red_squares" → "ρab")
    for color in _COLORS:
        s = s.replace(color, "")
    for marker in _MARKERS:
        s = s.replace(marker, "")

    # Remove any trailing/leading underscores or dashes that might remain
    s = s.strip("_- ")

    return s


def match_series_names(a_names: list[str], b_names: list[str]) -> list[tuple[str | None, str | None]]:
    """
    Match series names between Approach A and Approach B using normalized keys.

    Returns a list of (a_name, b_name) tuples. Unmatched names get None for the
    missing approach.
    """
    # Build normalized key -> original name maps
    a_norm = {normalize_series_name(n): n for n in a_names}
    b_norm = {normalize_series_name(n): n for n in b_names}

    matched = []
    matched_b_keys = set()

    for a_key, a_name in a_norm.items():
        if a_key in b_norm:
            matched.append((a_name, b_norm[a_key]))
            matched_b_keys.add(a_key)
        else:
            matched.append((a_name, None))

    # Add unmatched B names
    for b_key, b_name in b_norm.items():
        if b_key not in matched_b_keys:
            matched.append((None, b_name))

    return matched


def find_tc_inflection(temperatures, resistances):
    """
    Find Tc as the inflection point of the R(T) superconducting transition.
    Returns None if the series does not appear to be superconducting.

    Uses multiple smoothing factors and picks the most consistent result.
    Also falls back to simple midpoint if spline fitting fails.
    """
    t = np.array(temperatures, dtype=float)
    r = np.array(resistances, dtype=float)

    # Remove duplicates (same T, keep average R)
    unique_t, indices = np.unique(t, return_inverse=True)
    if len(unique_t) < len(t):
        avg_r = np.zeros(len(unique_t))
        counts = np.zeros(len(unique_t))
        for i, idx in enumerate(indices):
            avg_r[idx] += r[i]
            counts[idx] += 1
        avg_r /= counts
        t, r = unique_t, avg_r

    order = np.argsort(t)
    t, r = t[order], r[order]

    r_max = r.max()
    r_min = r.min()
    r_range = r_max - r_min

    if r_range == 0 or len(t) < 3:
        return None

    # Check if this series is superconducting (R drops to near zero)
    if r_min > 0.1 * r_max:
        return None

    # ── Identify transition region ──
    # Find where R crosses 10% and 90% of range
    threshold_10 = r_min + 0.10 * r_range
    threshold_90 = r_min + 0.90 * r_range

    # T where R first exceeds 10% (coming from low T)
    above_10 = np.where(r > threshold_10)[0]
    below_90 = np.where(r < threshold_90)[0]

    if len(above_10) == 0 or len(below_90) == 0:
        # Fallback: use full range
        t_trans_lo = t.min()
        t_trans_hi = t.max()
    else:
        # Transition starts where R first rises above 10% (from low T side)
        idx_lo = max(0, above_10[0] - 1)
        # Transition ends where R last stays below 90% (from low T side)
        idx_hi = min(len(t) - 1, below_90[-1] + 1)
        t_trans_lo = t[idx_lo]
        t_trans_hi = t[idx_hi]

    # Sanity: transition region shouldn't be wider than 60% of full T range
    t_full_range = t.max() - t.min()
    if t_trans_hi - t_trans_lo > 0.6 * t_full_range:
        t_trans_hi = t.min() + 0.5 * t_full_range
        t_trans_lo = t.min()

    # ── Method 1: Simple midpoint (T at R = 50% of R_range) ──
    r_mid = r_min + 0.5 * r_range
    # Interpolate to find T where R crosses r_mid
    tc_midpoint = None
    for i in range(len(t) - 1):
        if (r[i] - r_mid) * (r[i + 1] - r_mid) <= 0:
            # Linear interpolation
            if abs(r[i + 1] - r[i]) > 1e-12:
                frac = (r_mid - r[i]) / (r[i + 1] - r[i])
                tc_midpoint = t[i] + frac * (t[i + 1] - t[i])
            else:
                tc_midpoint = (t[i] + t[i + 1]) / 2
            break

    # ── Method 2: Spline derivative (inflection point) ──
    tc_spline = None
    if len(t) >= 4:
        # Try multiple smoothing factors
        n_pts = len(t)
        smooth_factors = [
            r_range * n_pts * 0.001,  # tight fit
            r_range * n_pts * 0.01,   # moderate
            r_range * n_pts * 0.1,    # smooth
        ]

        tc_candidates = []
        for sf in smooth_factors:
            try:
                spline = UnivariateSpline(t, r, s=sf)
                t_fine = np.linspace(t.min(), t.max(), 2000)
                dr_dt = spline.derivative()(t_fine)

                # Look for max |dR/dT| in transition region
                mask = (t_fine >= t_trans_lo) & (t_fine <= t_trans_hi)
                if mask.any():
                    dr_masked = np.abs(dr_dt[mask])
                    t_masked = t_fine[mask]
                    idx_peak = np.argmax(dr_masked)
                    tc_cand = float(t_masked[idx_peak])
                    tc_candidates.append(tc_cand)
            except Exception:
                continue

        if tc_candidates:
            # Use median to be robust against outliers
            tc_spline = float(np.median(tc_candidates))

    # ── Pick best result ──
    if tc_spline is not None and tc_midpoint is not None:
        # If spline and midpoint agree within 20% of transition width, use spline
        trans_width = t_trans_hi - t_trans_lo
        if trans_width > 0 and abs(tc_spline - tc_midpoint) < 0.5 * trans_width:
            return tc_spline
        else:
            # They disagree a lot — use midpoint (more robust)
            return tc_midpoint
    elif tc_spline is not None:
        return tc_spline
    elif tc_midpoint is not None:
        return tc_midpoint
    else:
        return None


def run_approach_a(client: ClaudeAPIClient, img_base64: str) -> dict:
    """Run Approach A (direct VLM Tc) and return parsed results."""
    response = client.vision_model_api_call(
        figure_base64=img_base64,
        prompt=DIRECT_TC_PROMPT,
        max_tokens=2048,
        temperature=0.0,
    )
    return parse_direct_tc_response(response)


def run_approach_b(client: ClaudeAPIClient, img_base64: str) -> dict:
    """Run Approach B (extract + compute) and return {series_name: tc_value or None}."""
    response = client.vision_model_api_call(
        figure_base64=img_base64,
        prompt=EXTRACT_RT_PROMPT,
        max_tokens=8192,
        temperature=0.0,
    )

    # Parse extraction
    _temp_extractor = ClaudeLinePlotDataExtractor.__new__(ClaudeLinePlotDataExtractor)
    extracted = _temp_extractor._parse_into_pydantic(response)

    results = {}
    for name, coords in extracted.name_to_coordinates.items():
        coords_arr = np.array(coords)
        if len(coords_arr) < 4:
            results[name] = None
            continue
        tc = find_tc_inflection(coords_arr[:, 0], coords_arr[:, 1])
        results[name] = tc  # float or None
    return results


# ─── Main ──────────────────────────────────────────────────────────────────

def main():
    desktop = "/Users/valeriegentzke/Desktop"
    # Files: try.png, try2.png, ..., try8.png
    image_files = []
    for name in ["try.png"] + [f"try{i}.png" for i in range(2, 9)]:
        path = os.path.join(desktop, name)
        if os.path.exists(path):
            image_files.append((name.replace(".png", ""), path))
        else:
            print(f"⚠ {name} not found, skipping")

    print(f"Found {len(image_files)} images: {[n for n, _ in image_files]}")

    output_csv = os.path.join(desktop, "tc_extraction_results.csv")
    rows = []

    client = ClaudeAPIClient("claude-sonnet-4-20250514")

    for img_name, img_path in image_files:
        print(f"\n{'='*60}")
        print(f"Processing: {img_name}")
        print(f"{'='*60}")

        img_base64 = load_image_base64(img_path)

        # ── Approach A ──
        print("  Running Approach A (direct VLM)...")
        try:
            a_results = run_approach_a(client, img_base64)
            print(f"  → Found {len(a_results)} series")
        except Exception as e:
            print(f"  ✗ Approach A failed: {e}")
            traceback.print_exc()
            a_results = {}

        # ── Approach B ──
        print("  Running Approach B (extract + compute)...")
        try:
            b_results = run_approach_b(client, img_base64)
            print(f"  → Found {len(b_results)} series")
        except Exception as e:
            print(f"  ✗ Approach B failed: {e}")
            traceback.print_exc()
            b_results = {}

        # ── Match series names between A and B ──
        matched_pairs = match_series_names(
            list(a_results.keys()),
            list(b_results.keys()),
        )

        for a_name, b_name in matched_pairs:
            # Use Approach A name if available (more human-readable), else B
            display_name = a_name if a_name else b_name

            # Approach A values
            a_vals = a_results.get(a_name, {}) if a_name else {}
            a_sc = a_vals.get("superconducting", None)
            if a_sc is True:
                tc_a = a_vals.get("tc_mid")
                tc_a_str = f"{tc_a:.1f}" if tc_a is not None else "NA"
            elif a_sc is False:
                tc_a_str = "NA"
            else:
                tc_a_str = ""  # not found in Approach A

            # Approach B values
            tc_b = b_results.get(b_name) if b_name else None
            b_found = b_name in b_results if b_name else False
            if tc_b is not None:
                tc_b_str = f"{tc_b:.1f}"
            elif b_found:
                tc_b_str = "NA"  # found but not SC
            else:
                tc_b_str = ""  # not found in Approach B

            # Superconducting status (prefer Approach A classification)
            if a_sc is True:
                sc_status = "YES"
            elif a_sc is False:
                sc_status = "NO"
            elif tc_b is not None:
                sc_status = "YES"
            elif b_found and tc_b is None:
                sc_status = "NO"
            else:
                sc_status = "?"

            row = {
                "source": img_name,
                "material": display_name,
                "Tc_approach_A": tc_a_str,
                "Tc_approach_B": tc_b_str,
                "Tc_hand_annotated": "",  # to be filled manually
                "Tc_text_extracted": "",  # to be filled manually
                "superconducting": sc_status,
            }
            rows.append(row)
            match_info = f"(A={a_name}, B={b_name})" if a_name and b_name else ""
            print(f"    {display_name}: A={tc_a_str}, B={tc_b_str}, SC={sc_status} {match_info}")

    # ── Write CSV ──
    fieldnames = [
        "source",
        "material",
        "Tc_approach_A",
        "Tc_approach_B",
        "Tc_hand_annotated",
        "Tc_text_extracted",
        "superconducting",
    ]

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{'='*60}")
    print(f"Done! Results written to: {output_csv}")
    print(f"Total: {len(rows)} rows from {len(image_files)} images")
    print(f"API cost so far: ${client.get_cost():.4f}")


if __name__ == "__main__":
    main()
