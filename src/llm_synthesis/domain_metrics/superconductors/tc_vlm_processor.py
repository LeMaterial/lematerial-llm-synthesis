"""Extract Tc from R(T) plots via VLM (geometric construction).

Implements BaseVLMMetricProcessor for the superconductivity domain.
All prompt text, parsing logic, and sanity-check heuristics are lifted
directly from batch_run_tc.py to preserve extraction quality.
"""

import re
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
inset_tc_values: <series: value K, series: value K, ... (if
    tc_summary type), otherwise "N/A">
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
    """Build the VLM Tc extraction prompt, optionally anchoring series names."""
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
    return DIRECT_TC_PROMPT_TEMPLATE.format(series_name_instruction=instruction)


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
    """Parse the structured text output of the VLM Tc prompt."""
    results: dict[str, dict] = {}
    current: str | None = None
    inset_tc_raw: str | None = None

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


def sanity_check_delta_tc(vlm_results: dict[str, dict]) -> dict[str, dict]:
    """Apply Delta_Tc sanity check and inset-override logic.

    Rules (matching original batch_run_tc.py exactly):
    - If VLM says not SC but an inset Tc exists → override to SC with inset
      value.
    - If Delta_Tc > 10 K and an inset Tc exists → use inset value instead.
    - If Delta_Tc > 10 K and no inset → mark as not SC (likely metallic
      decline).
    - If geometric Tc differs from inset by >20% → use inset value.
    """
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
    """Extract Tc from R(T) plots via geometric construction (Claude VLM).

    Args:
        claude_model: Claude model name to use for the VLM call.
        max_tokens: Max tokens for the VLM response.
    """

    def __init__(
        self,
        claude_model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 8000,
    ) -> None:
        self._model = claude_model
        self._max_tokens = max_tokens

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
        from llm_synthesis.services.llm_api.claude import ClaudeAPIClient

        client = ClaudeAPIClient(self._model)
        tc_from_vlm: dict[int, dict[str, dict]] = {}

        for idx, plot in relevant_plots:
            fig = plot_figures[idx]
            known_series = list(plot.name_to_coordinates.keys())
            prompt = build_tc_prompt(known_series_names=known_series)
            try:
                response = client.vision_model_api_call(
                    figure_base64=fig.base64_data,
                    prompt=prompt,
                    max_tokens=self._max_tokens,
                    temperature=0.0,
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
