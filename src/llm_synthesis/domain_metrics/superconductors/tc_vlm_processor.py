"""Extract Tc from R(T) plots via VLM (geometric construction).

Implements BaseVLMMetricProcessor for the superconductivity domain.
All prompt text, parsing logic, and sanity-check heuristics are lifted
directly from batch_run_tc.py to preserve extraction quality.
"""

import re
import threading
from typing import Any

from llm_synthesis.domain_metrics.base import BaseVLMMetricProcessor
from llm_synthesis.models.figure import FigureInfo
from llm_synthesis.models.performance import PlotMaterialMapping
from llm_synthesis.models.plot import ExtractedLinePlotData
from llm_synthesis.utils.formula_utils import normalize_formula

# ---------------------------------------------------------------------------
# Tc VLM prompt
# ---------------------------------------------------------------------------

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
  - Resistance drops from a finite value all the way to ZERO
    (or very close to zero)
  - It looks like a cliff or step function, not a gradual slope

If you see a curve that gradually decreases over 10-50 K, that is NOT
superconductivity — it is normal metallic/Kondo behavior.

STEP 0 — EXAMINE THE FULL FIGURE (main plot + any insets/panels):
a) Identify ALL panels in the figure: the main plot and any insets, secondary
   panels, or embedded sub-plots. For each one, describe:
     - What quantity is on each axis (e.g., ρ vs T, Tc vs x,
       phase diagram, etc.)
     - The axis ranges and tick marks
     - Whether it contains information relevant to determining Tc

b) CATEGORIZE each inset/panel into one of these types:
     (i)   ZOOMED R(T): A magnified view of the transition region of the same
           R(T) data as the main plot. Has the same axes (R vs T) but a narrower
           temperature range. → Use this PREFERENTIALLY for geometric Tc
           construction (much better spatial resolution).
     (ii)  Tc SUMMARY: Shows Tc (or T_onset, T_zero) as a function
            of composition, pressure, doping, field, etc.
            (e.g., "Tc vs x" or a phase diagram with
           a Tc boundary). → Read Tc values DIRECTLY from this panel for each
           series/composition. These are typically the most
            accurate values available.
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

STEP 0.7 — DETECT CONDITION-VARYING PLOTS:
Check if the different series in this plot represent the SAME material
measured under different external conditions (magnetic field H, pressure P,
etc.) rather than different compositions.

Signs of a condition-varying plot:
- Legend entries differ only by a field/pressure value (e.g., "0 T", "1 T", "5 T")
- Legend uses "H = ...", "B = ...", "P = ...", "GPa" labels
- All series are the same material formula with only field/pressure changing

If this is a condition-varying plot:
- Report: condition_variable: <magnetic_field|pressure|none>
- Extract Tc ONLY for the zero-field / ambient-pressure series
- Do NOT create separate entries for each field/pressure value
- Do NOT append field/pressure values to material names

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

STEP 4.5 — IDENTIFY THE MATERIAL FOR EACH SERIES:
{material_instruction}

IMPORTANT: NEVER append magnetic field values (T, Tesla, Oe), pressure values
(GPa, kbar), or other measurement conditions to the material name. The material
name should contain ONLY the chemical formula/composition.

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

condition_variable: <magnetic_field|pressure|none>
inset_detected: <yes/no>
inset_type: <"zoomed_rt" | "tc_summary" | "other" | "none">
inset_description: <brief description of what the inset shows, or "N/A">
inset_axes: <tick marks of inset if present, otherwise "N/A">
inset_tc_values: <series: value K, series: value K, ... (if
    tc_summary type), otherwise "N/A">
main_x_axis_ticks: <list>
main_y_axis_ticks: <list>

Series: <name>
material: <matched material name from the provided list, or "unknown">
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

# Two-image variant: full figure + a zoomed bottom-left crop (low-T, low-R
# region, ~4x resolution there). Ported from batch_run_tc_new_snippet.py's
# PROMPT_TEMPLATE_SNIPPET -- gives the VLM a second, higher-resolution look
# at the transition region, which the single-image prompt above misses on
# figures where the SC transition is a small feature in a busy full plot.
SNIPPET_TC_PROMPT_TEMPLATE = """
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
   zoomed view. These are CRITICAL for precise Tc determination.

   CROP VALIDATION: Verify that Image 2 actually shows the bottom-left
   region of Image 1. If the crop does NOT match (wrong axis labels, wrong
   scale, or a different panel/inset), IGNORE Image 2 entirely and use
   only Image 1 for all analysis.

c) CATEGORIZE each inset/panel into:
     (i)   ZOOMED R(T) -> Use PREFERENTIALLY for geometric Tc construction.
     (ii)  Tc SUMMARY  -> Read Tc values DIRECTLY.
     (iii) OTHER        -> Note but do not use.

d) SCALE AWARENESS: Compare tick marks from Image 1 vs Image 2. If the full
   plot spans a wide range but the crop shows clear tick marks, USE THE CROP
   for reading transition temperatures. If the transition falls outside the
   crop, use the full image.

STEP 0.5 — EXTRACT Tc FROM SUMMARY INSETS (if any type-(ii) inset found):
Read Tc values directly:
  inset_tc_<series_name>: <value> K
These serve as a REFERENCE for Step 4 cross-check.

STEP 0.7 — DETECT CONDITION-VARYING PLOTS:
Check if the different series in this plot represent the SAME material
measured under different external conditions (magnetic field H, pressure P,
etc.) rather than different compositions.

Signs of a condition-varying plot:
- Legend entries differ only by a field/pressure value (e.g., "0 T", "1 T", "5 T")
- Legend uses "H = ...", "B = ...", "P = ...", "GPa" labels
- All series are the same material formula with only field/pressure changing

If this is a condition-varying plot:
- Report: condition_variable: <magnetic_field|pressure|none>
- Extract Tc ONLY for the zero-field / ambient-pressure series
- Do NOT create separate entries for each field/pressure value
- Do NOT append field/pressure values to material names

STEP 1 — IDENTIFY SERIES:
{series_name_instruction}
List every distinct curve visible in the plot.

STEP 1.5 — IDENTIFY THE MATERIAL FOR EACH SERIES:
{material_instruction}

IMPORTANT: NEVER append magnetic field values (T, Tesla, Oe), pressure values
(GPa, kbar), or other measurement conditions to the material name. The material
name should contain ONLY the chemical formula/composition.

STEP 2 — READ RESISTANCE VALUES AT LOWEST AND HIGHEST TEMPERATURE:
For EACH series, use Image 2 (crop) for low-T values if the series is
visible there:
  a) R_at_lowest_T: resistance at the LOWEST temperature
  b) R_at_highest_T: resistance at the HIGHEST temperature

STEP 3 — CONFIRM SUPERCONDUCTIVITY:
Superconducting ONLY if R_at_lowest_T is approximately zero.

STRICT ZERO CHECK (use Image 2 crop for better precision):
In the cropped view, check whether the series data points at low T are
truly sitting ON the y=0 line. If a series has R clearly above zero in the
crop — even if it looked close to zero in the full image — it is NOT
superconducting.

STEP 4 — GEOMETRIC Tc CONSTRUCTION (only for confirmed superconductors):
Use the best view — in order of preference:
  1. A zoomed inset (if present in the figure)
  2. Image 2 (bottom-left crop) — if the transition is visible there
  3. Image 1 (full plot) — only if the transition is outside the crop

SCAN FROM HIGH T TO LOW T (right to left on the plot):
  a) R_normal: the resistance value on the plateau IMMEDIATELY BEFORE the
     sharp superconducting drop — NOT the maximum resistance at 300 K.
  b) T_onset: Scanning LEFT from R_normal, the temperature where resistance
     FIRST begins to drop sharply below the normal-state plateau.
  c) T_zero: Continuing LEFT, the temperature where resistance FIRST
     reaches approximately zero. This is NOT the lowest temperature in the
     dataset — it is where the transition ENDS.
  d) SANITY CHECK: T_onset - T_zero should typically be 0.5-5 K. If > 10 K,
     you are probably measuring metallic decline, NOT the SC transition.
  e) Tc_mid = (T_onset + T_zero) / 2.
  f) Delta_Tc = T_onset - T_zero
  g) CROSS-CHECK: If inset Tc differs by >20%, use inset value.

STEP 5 — RELATIVE ORDERING OF TRANSITIONS:
Compare which series transitions at higher vs lower T.

Output format:

condition_variable: <magnetic_field|pressure|none>
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
material: <matched material name from the provided list, or "unknown">
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


def crop_bottom_left_quadrant(figure_base64: str) -> str:
    """Crop bottom-left quadrant of a base64 image -> base64 PNG.

    ~4x resolution in the SC transition region (low T, low R).
    """
    import base64
    import io

    from PIL import Image as PILImage

    img_bytes = base64.b64decode(figure_base64)
    img = PILImage.open(io.BytesIO(img_bytes))
    w, h = img.size
    cropped = img.crop((0, h // 2, w // 2, h))
    buf = io.BytesIO()
    cropped.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def build_tc_prompt(
    known_series_names: list[str] | None = None,
    materials: list[str] | None = None,
) -> str:
    """Build the VLM Tc extraction prompt, optionally anchoring series names
    and the list of candidate materials to match each series against."""
    if known_series_names:
        names_list = "\n".join(
            f"  {i + 1}. {name}" for i, name in enumerate(known_series_names)
        )
        instruction = (
            "The following series were previously identified in this plot. "
            "You MUST use these EXACT names in your output "
            "(in the 'Series:' lines):\n"
            f"{names_list}\n\n"
            "If you see additional curves not in this list, "
            "add them with descriptive names."
        )
    else:
        instruction = (
            "List every distinct curve (by legend label, color, marker)."
        )

    if materials:
        mat_list = "\n".join(f"  - {m}" for m in materials)
        material_instruction = (
            "This paper studies the following material compositions:\n"
            f"{mat_list}\n\n"
            "For each series/curve, use the legend label, color, and "
            "surrounding figure caption/context to match it to EXACTLY ONE "
            "of the materials above. Report the matched material's exact "
            "name (as written above) in the 'material:' line. If you "
            "cannot confidently match a series to any listed material, "
            "write 'unknown'."
        )
    else:
        material_instruction = (
            "No candidate material list was provided — report your best "
            "guess of the material composition from the legend/caption in "
            "the 'material:' line, or 'unknown' if unclear."
        )

    return DIRECT_TC_PROMPT_TEMPLATE.format(
        series_name_instruction=instruction,
        material_instruction=material_instruction,
    )


def build_snippet_tc_prompt(
    known_series_names: list[str] | None = None,
    materials: list[str] | None = None,
) -> str:
    """Two-image (full + bottom-left crop) variant of build_tc_prompt."""
    if known_series_names:
        names_list = "\n".join(
            f"  {i + 1}. {name}" for i, name in enumerate(known_series_names)
        )
        instruction = (
            "The following series were previously identified in this plot. "
            "You MUST use these EXACT names in your output "
            "(in the 'Series:' lines):\n"
            f"{names_list}\n\n"
            "If you see additional curves not in this list, "
            "add them with descriptive names."
        )
    else:
        instruction = (
            "List every distinct curve (by legend label, color, marker)."
        )

    if materials:
        mat_list = "\n".join(f"  - {m}" for m in materials)
        material_instruction = (
            "This paper studies the following material compositions:\n"
            f"{mat_list}\n\n"
            "For each series/curve, use the legend label, color, and "
            "surrounding figure caption/context to match it to EXACTLY ONE "
            "of the materials above. Report the matched material's exact "
            "name (as written above) in the 'material:' line. If you "
            "cannot confidently match a series to any listed material, "
            "write 'unknown'."
        )
    else:
        material_instruction = (
            "No candidate material list was provided — report your best "
            "guess of the material composition from the legend/caption in "
            "the 'material:' line, or 'unknown' if unclear."
        )

    return SNIPPET_TC_PROMPT_TEMPLATE.format(
        series_name_instruction=instruction,
        material_instruction=material_instruction,
    )


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def _parse_inset_tc_values(raw: str, known_series: dict) -> dict[str, float]:
    result: dict[str, float] = {}
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry or ":" not in entry:
            continue
        name_part, val_part = entry.rsplit(":", 1)
        name_part = name_part.strip()
        val_part = val_part.strip()
        match = re.search(r"(\d+\.?\d*|\d*\.\d+)", val_part)
        if not match:
            continue
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


def parse_direct_tc_response(response_text: str) -> dict[str, dict]:
    """Parse the structured text output of the VLM Tc prompt.

    Returns a dict keyed by series name. A top-level metadata key
    ``"__condition_variable__"`` is added with value ``"magnetic_field"``,
    ``"pressure"``, or ``"none"`` when present in the response.
    """
    results: dict[str, dict] = {}
    current: str | None = None
    inset_tc_raw: str | None = None
    condition_variable: str = "none"

    for line in response_text.strip().split("\n"):
        line = line.strip()
        # Parse condition_variable (appears before any Series: block)
        if line.lower().startswith("condition_variable:") and current is None:
            condition_variable = line.split(":", 1)[1].strip().lower()
            continue
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
            elif key in ("reason", "source", "material"):
                results[current][key] = val
            else:
                # These are all numeric fields downstream (t_onset, t_zero,
                # tc_mid, delta_tc, inset_tc, ...) -- always coerce to
                # float|None, never leave a raw string that would crash a
                # later `> 0` / `> 10` comparison.
                match = re.search(r"(\d+\.?\d*|\d*\.\d+)", val)
                if match:
                    try:
                        results[current][key] = float(match.group(1))
                    except ValueError:
                        results[current][key] = None
                else:
                    results[current][key] = None

    if inset_tc_raw and inset_tc_raw.lower() not in ("n/a", "none", ""):
        inset_pairs = _parse_inset_tc_values(inset_tc_raw, results)
        for series_name, tc_val in inset_pairs.items():
            if series_name in results:
                results[series_name]["inset_tc"] = tc_val

    # Store condition_variable as metadata (won't collide with series names)
    results["__condition_variable__"] = {"value": condition_variable}

    return results


def filter_condition_varying_series(vlm_results: dict[str, dict]) -> dict[str, dict]:
    """Remove non-ambient series when the plot varies field or pressure.

    If ``condition_variable`` is ``magnetic_field`` or ``pressure``, only
    keep series whose name does NOT contain a field/pressure label
    (e.g. "1 T", "5 GPa"). This eliminates duplicate entries where the
    VLM created separate "materials" for each measurement condition.
    """
    meta = vlm_results.pop("__condition_variable__", None)
    condition = (meta or {}).get("value", "none")

    if condition == "none":
        return vlm_results

    # Patterns indicating a non-ambient condition in the series name
    _FIELD_PATTERN = re.compile(
        r"(?:\d+\.?\d*)\s*(?:T|Tesla|Oe|kOe)\b", re.IGNORECASE
    )
    _PRESSURE_PATTERN = re.compile(
        r"(?:\d+\.?\d*)\s*(?:GPa|kbar|Mbar)\b", re.IGNORECASE
    )

    pattern = (
        _FIELD_PATTERN if condition == "magnetic_field" else _PRESSURE_PATTERN
    )

    filtered: dict[str, dict] = {}
    for series_name, vals in vlm_results.items():
        if pattern.search(series_name):
            continue  # skip non-ambient series
        filtered[series_name] = vals

    # If filtering removed everything, return original (safety fallback)
    if not filtered:
        return vlm_results

    return filtered


def sanity_check_delta_tc(vlm_results: dict[str, dict]) -> dict[str, dict]:
    """Apply Delta_Tc sanity check and inset-override logic.

    Rules (matching original batch_run_tc.py exactly):
    - If VLM says not SC but an inset Tc exists → override to SC with inset
      value.
    - If Delta_Tc > 10 K and an inset Tc exists → use inset value instead.
    - If Delta_Tc > 10 K and no inset → mark as not SC (likely metallic
      decline).
    - If geometric Tc differs from inset by >20% → use inset value.

    Also applies condition-varying-plot filtering (field/pressure exclusion).
    """
    # First filter out non-ambient series if condition_variable is set
    vlm_results = filter_condition_varying_series(vlm_results)

    corrected: dict[str, dict] = {}
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
                    f"VLM said SC=NO, but inset Tc={inset_tc:.1f} K found. "
                    "Using inset value."
                )
            continue

        delta = vals.get("delta_tc")
        if delta is not None and delta > 10:
            inset_tc = vals.get("inset_tc")
            if inset_tc is not None and inset_tc > 0:
                corrected[series_name]["tc_mid"] = inset_tc
                corrected[series_name]["source"] = "inset"
                corrected[series_name]["_sanity_override"] = (
                    f"Delta_Tc={delta:.1f} K too wide (>10 K). "
                    f"Using inset Tc={inset_tc:.1f} K instead."
                )
                for k in ("t_onset", "t_zero", "delta_tc"):
                    corrected[series_name].pop(k, None)
            else:
                corrected[series_name]["superconducting"] = False
                corrected[series_name]["_sanity_override"] = (
                    f"Delta_Tc={delta:.1f} K is too wide (>10 K). "
                    "Likely confusing metallic decline with SC transition."
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
                    f"Geometric Tc_mid={tc_mid:.1f} K differs from "
                    f"inset Tc={inset_tc:.1f} K by "
                    f"{relative_diff * 100:.0f}% (>20%). Using inset value."
                )

    return corrected


# ---------------------------------------------------------------------------
# Series → material fuzzy matching
# ---------------------------------------------------------------------------


def _find_vlm_tc_for_series(
    series_name: str,
    vlm_results: dict[str, dict],
) -> dict:
    """Fuzzy-match a series name to a VLM result entry."""
    if series_name in vlm_results:
        return vlm_results[series_name]

    sn = normalize_formula(series_name)
    for vlm_key, vlm_val in vlm_results.items():
        vk = normalize_formula(vlm_key)
        if sn in vk or vk in sn:
            return vlm_val
        sn_parts = re.split(r"[\s_-]+", series_name.lower())
        vk_parts = re.split(r"[\s_-]+", vlm_key.lower())
        if (
            len(sn_parts) >= 1
            and len(vk_parts) >= 1
            and sn_parts[0] == vk_parts[0]
        ):
            return vlm_val

    if len(vlm_results) == 1:
        return next(iter(vlm_results.values()))
    return {}


def _vlm_data_has_tc(vlm_data: dict) -> bool:
    return bool(
        vlm_data.get("superconducting") and vlm_data.get("tc_mid") is not None
    )


# ---------------------------------------------------------------------------
# BaseVLMMetricProcessor implementation
# ---------------------------------------------------------------------------


class TcVLMProcessor(BaseVLMMetricProcessor):
    """Extract Tc from R(T) plots via geometric construction (any LiteLLM VLM).

    Args:
        claude_model: LLM_REGISTRY key or raw litellm model string
            (e.g. "qwen3.5-397b-a17b", "claude-sonnet-4.6").
            Kept the name ``claude_model`` for backwards compatibility with
            existing DomainConfig callers.
        max_tokens: Max tokens for the VLM response.
    """

    def __init__(
        self,
        claude_model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 8000,
    ) -> None:
        self._model = claude_model
        self._max_tokens = max_tokens
        self._cumulative_cost_usd = 0.0
        self._cost_lock = threading.Lock()

    def get_cost(self) -> float:
        return self._cumulative_cost_usd

    def _vision_call(self, figure_base64: str, prompt: str) -> str:
        """Single-image vision call via litellm."""
        return self._vision_call_multi([figure_base64], prompt)

    def _vision_call_multi(self, figure_base64s: list[str], prompt: str) -> str:
        """Vision call via litellm with one or more images, resolving
        LLM_REGISTRY if applicable."""
        import litellm

        from llm_synthesis.utils.llms import LLM_REGISTRY

        model = self._model
        kwargs: dict[str, Any] = {}
        if model in LLM_REGISTRY.configs:
            cfg = LLM_REGISTRY.configs[model]
            model = cfg.model
            if cfg.api_key:
                kwargs["api_key"] = cfg.api_key
            if cfg.api_base:
                kwargs["api_base"] = cfg.api_base
            for k, v in (cfg.extra_kwargs or {}).items():
                if k not in ("thinking", "reasoning_effort", "enable_thinking"):
                    kwargs[k] = v

        content: list[dict[str, Any]] = []
        for b64 in figure_base64s:
            image_type = "jpeg" if b64.startswith("/9j/") else "png"
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/{image_type};base64,{b64}"
                    },
                }
            )
        content.append({"type": "text", "text": prompt})

        response = litellm.completion(
            model=model,
            max_tokens=self._max_tokens,
            temperature=0.0,
            messages=[{"role": "user", "content": content}],
            **kwargs,
        )
        try:
            cost = litellm.completion_cost(completion_response=response)
            with self._cost_lock:
                self._cumulative_cost_usd += cost
        except Exception:
            pass
        result = response.choices[0].message.content
        if result is None:
            # e.g. reasoning tokens exhausted max_tokens before any visible
            # output -- treat as a failed call, not a value to parse.
            finish_reason = getattr(response.choices[0], "finish_reason", "?")
            raise ValueError(
                f"VLM returned empty content (finish_reason={finish_reason})"
            )
        return result

    def process(
        self,
        relevant_plots: list[tuple[int, ExtractedLinePlotData]],
        plot_figures: list[FigureInfo],
        plot_mappings: list[PlotMaterialMapping],
        materials: list[str],
        paper_text: str,
    ) -> dict[str, Any]:
        """Run VLM Tc extraction on every relevant R(T) plot.

        Returns per-material dict with keys matching tc_from_vlm in
        the original batch_run_tc.py output format.
        """
        tc_from_vlm: dict[int, dict[str, dict]] = {}

        for idx, plot in relevant_plots:
            fig = plot_figures[idx]
            known_series = list(plot.name_to_coordinates.keys())
            prompt = build_tc_prompt(known_series_names=known_series)
            try:
                response = self._vision_call(
                    figure_base64=fig.base64_data,
                    prompt=prompt,
                )
                parsed = parse_direct_tc_response(response)
                corrected = sanity_check_delta_tc(parsed)
                tc_from_vlm[idx] = corrected
            except Exception as e:
                import logging

                logging.getLogger(__name__).warning(
                    "TcVLMProcessor: plot %d failed: %s", idx, e
                )

        # Map VLM results to materials via plot_mappings
        vlm_tc_per_material: dict[str, dict] = {}
        for mapping in plot_mappings:
            plot_idx = mapping.plot_index
            if plot_idx not in tc_from_vlm:
                continue
            vlm_results_for_plot = tc_from_vlm[plot_idx]
            for sm in mapping.mappings:
                vlm_data = _find_vlm_tc_for_series(
                    sm.series_name, vlm_results_for_plot
                )
                if vlm_data:
                    existing = vlm_tc_per_material.get(sm.material_name)
                    if existing is None:
                        vlm_tc_per_material[sm.material_name] = vlm_data
                    elif _vlm_data_has_tc(vlm_data) and not _vlm_data_has_tc(
                        existing
                    ):
                        vlm_tc_per_material[sm.material_name] = vlm_data

        # Convert to the output format expected by the batch runner / writer
        results: dict[str, Any] = {}
        for material, vlm_entry in vlm_tc_per_material.items():
            results[material] = {
                "superconducting": vlm_entry.get("superconducting"),
                "T_onset": vlm_entry.get("t_onset"),
                "Tc_mid": vlm_entry.get("tc_mid"),
                "T_zero": vlm_entry.get("t_zero"),
                "Delta_Tc": vlm_entry.get("delta_tc"),
                "source": vlm_entry.get("source", "main plot"),
            }
        return results

    def _process_one_figure(
        self, fig: FigureInfo, materials: list[str], use_snippet: bool = True
    ) -> dict[str, dict] | None:
        import logging

        if use_snippet:
            try:
                crop_b64 = crop_bottom_left_quadrant(fig.base64_data)
                prompt = build_snippet_tc_prompt(materials=materials)
                response = self._vision_call_multi(
                    [fig.base64_data, crop_b64], prompt
                )
                parsed = parse_direct_tc_response(response)
                return sanity_check_delta_tc(parsed)
            except Exception as e:
                logging.getLogger(__name__).warning(
                    "TcVLMProcessor: figure %s snippet call failed (%s) --"
                    " falling back to single-image",
                    fig.figure_reference,
                    e,
                )
                # fall through to single-image path below

        prompt = build_tc_prompt(materials=materials)
        try:
            response = self._vision_call(
                figure_base64=fig.base64_data,
                prompt=prompt,
            )
            parsed = parse_direct_tc_response(response)
            return sanity_check_delta_tc(parsed)
        except Exception as e:
            logging.getLogger(__name__).warning(
                "TcVLMProcessor: figure %s failed: %s",
                fig.figure_reference,
                e,
            )
            return None

    def process_from_figures(
        self,
        figures: list[FigureInfo],
        materials: list[str],
        max_workers: int = 8,
        use_snippet: bool = True,
    ) -> dict[str, Any]:
        """Read Tc directly from figures, no digitization/linking pass.

        One VLM call per figure does both jobs at once: reads Tc via
        geometric construction AND matches each curve to one of
        ``materials`` using the legend/caption/context visible in the
        image. Skips extract_plot_data + link_performance entirely.
        Figures are processed concurrently (independent VLM calls).

        use_snippet: send both the full figure AND a zoomed bottom-left
            crop (low-T, low-R region) in the same call -- ported from the
            original Claude pipeline's crop-retry step, which gave the VLM
            a second, higher-resolution look at the transition region.
            Falls back to single-image per figure if cropping/the two-image
            call fails.
        """
        from concurrent.futures import ThreadPoolExecutor

        vlm_tc_per_material: dict[str, dict] = {}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for corrected in executor.map(
                lambda fig: self._process_one_figure(
                    fig, materials, use_snippet=use_snippet
                ),
                figures,
            ):
                if not corrected:
                    continue
                for series_name, vlm_data in corrected.items():
                    material = vlm_data.get("material")
                    if not material or material.lower() == "unknown":
                        continue
                    existing = vlm_tc_per_material.get(material)
                    if existing is None or (
                        _vlm_data_has_tc(vlm_data)
                        and not _vlm_data_has_tc(existing)
                    ):
                        vlm_tc_per_material[material] = vlm_data

        results: dict[str, Any] = {}
        for material, vlm_entry in vlm_tc_per_material.items():
            results[material] = {
                "superconducting": vlm_entry.get("superconducting"),
                "T_onset": vlm_entry.get("t_onset"),
                "Tc_mid": vlm_entry.get("tc_mid"),
                "T_zero": vlm_entry.get("t_zero"),
                "Delta_Tc": vlm_entry.get("delta_tc"),
                "source": vlm_entry.get("source", "main plot"),
            }
        return results
