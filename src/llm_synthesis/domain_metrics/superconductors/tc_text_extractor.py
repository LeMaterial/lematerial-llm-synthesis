"""Extract superconducting critical temperatures (Tc) from paper text.

Implements BaseTextMetricExtractor for the superconductivity domain.
Logic lifted directly from batch_run_tc.py to preserve extraction quality.
"""

import re
from typing import Any

import dspy

from llm_synthesis.domain_metrics.base import BaseTextMetricExtractor
from llm_synthesis.transformers.material_extraction.dspy_extraction import (
    DspyTextExtractor,
    make_dspy_text_extractor_signature,
)
from llm_synthesis.utils.formula_utils import normalize_formula

_MAX_TEXT_CHARS = 60_000

_TC_TEXT_INSTRUCTIONS = (
    "Extract ALL critical temperature (Tc) values "
    "reported in this superconductivity paper. "
    "For EACH material that has a Tc value mentioned "
    "in the text, report:\n"
    "  - The material formula\n"
    "  - T_onset (if explicitly reported)\n"
    "  - Tc: the critical temperature\n"
    "  - T_zero (if explicitly reported)\n"
    "  - Whether the material is superconducting (YES/NO)\n\n"
    "Only extract values explicitly stated. "
    "Use 'NR' for values not reported."
)

_TC_TEXT_INPUT_DESCRIPTION = (
    "The full publication text from a superconductivity paper."
)

_TC_TEXT_OUTPUT_DESCRIPTION = (
    "For each material, one line in the format:\n"
    "material_formula | superconducting: YES/NO | "
    "T_onset: <value> K | Tc: <value> K | "
    "T_zero: <value> K\n"
    "Use 'NR' for values not reported."
)


def _parse_tc_text_response(raw_text: str) -> dict[str, dict]:
    """Parse the pipe-delimited text output from the TextToTc LLM call."""
    results: dict[str, dict] = {}
    for line in raw_text.strip().split("\n"):
        line = line.strip()
        if not line or "|" not in line:
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 2:
            continue
        material = parts[0].strip()
        entry: dict[str, Any] = {
            "superconducting": False,
            "T_onset": None,
            "Tc_mid": None,
            "T_zero": None,
        }
        for part in parts[1:]:
            part_lower = part.lower().strip()
            if "superconducting" in part_lower:
                entry["superconducting"] = "yes" in part_lower
            else:
                match = re.match(
                    r"(t_onset|tc_mid|tc|t_zero)\s*:\s*(\d+\.?\d*)\s*k?",
                    part_lower,
                )
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


def _find_matching_text_tc(
    material: str,
    tc_from_text: dict[str, dict],
) -> list[tuple[str, dict]]:
    """Fuzzy-match a material name against the Tc-from-text dict."""
    mat_norm = normalize_formula(material)
    matches = []
    for tc_key, tc_entry in tc_from_text.items():
        key_norm = normalize_formula(tc_key)
        if key_norm == mat_norm:
            matches.append((tc_key, tc_entry))
    matches.sort(
        key=lambda x: (
            not x[1].get("superconducting", False),
            -(x[1].get("Tc_mid") or 0),
        )
    )
    return matches


class TcTextExtractor(BaseTextMetricExtractor):
    """Extract Tc values mentioned explicitly in paper text.

    Uses a DSPy text extractor with a specialized TextToTc signature,
    then fuzzy-matches results to the material list.

    Args:
        lm: DSPy LM to use.  Typically gemini-3.0-flash at temperature 0.
    """

    def __init__(self, lm: dspy.LM) -> None:
        sig = make_dspy_text_extractor_signature(
            signature_name="TextToTc",
            instructions=_TC_TEXT_INSTRUCTIONS,
            input_description=_TC_TEXT_INPUT_DESCRIPTION,
            output_name="tc_values",
            output_description=_TC_TEXT_OUTPUT_DESCRIPTION,
        )
        self._extractor = DspyTextExtractor(signature=sig, lm=lm)

    def extract(
        self,
        paper_text: str,
        materials: list[str],
    ) -> dict[str, Any]:
        """Run Tc text extraction and return per-material Tc dicts."""
        truncated = (
            paper_text[:_MAX_TEXT_CHARS]
            if len(paper_text) > _MAX_TEXT_CHARS
            else paper_text
        )
        try:
            raw = self._extractor.forward(input=truncated)
        except Exception:
            raw = ""

        tc_from_text = _parse_tc_text_response(raw)

        results: dict[str, Any] = {}
        for material in materials:
            matches = _find_matching_text_tc(material, tc_from_text)
            if matches:
                best_key, best_entry = matches[0]
                entry = dict(best_entry)
                if len(matches) > 1:
                    entry["_all_variants"] = [
                        {"condition": k, **v} for k, v in matches
                    ]
                results[material] = entry
        return results
