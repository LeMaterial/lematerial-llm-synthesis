"""Utility functions for chemical formula normalization and matching."""

import re


def normalize_formula(s: str) -> str:
    """Normalize a chemical formula string for fuzzy matching.

    Handles LaTeX subscripts (_{0.12}), superscripts (^{...}),
    LaTeX commands (\\mathrm{}, \\text{}, etc.),
    unicode subscript digits, greek letters,
    and parenthetical suffixes like '(C)' or '(NC)'.

    Returns a lowercase, whitespace-stripped string suitable for
    equality comparison.
    """
    base = s.strip()
    # Strip LaTeX text-mode commands: \mathrm{X} -> X
    latex_cmd = r"\\(?:mathrm|text|textit|mathit|mathbf)\{([^}]*)\}"
    base = re.sub(latex_cmd, r"\1", base)
    # Strip LaTeX sub/superscripts: _{0.12} -> 0.12
    base = re.sub(r"[_^]\{([^}]*)\}", r"\1", base)
    # Remove $ and remaining backslashes
    base = base.replace("$", "").replace("\\", "")
    # Strip trailing parenthetical annotations: (C), (NC), (centrosymmetric),
    # etc.
    base = re.sub(r"\s*\([^)]*\)\s*$", "", base).strip()
    # Strip trailing bracket annotations: [dashed-dotted line], etc.
    base = re.sub(r"\s*\[[^\]]*\]\s*$", "", base).strip()
    # Greek letter normalization
    base = base.replace("\u03b4", "delta").replace("\u0394", "delta")
    # Unicode subscript digits
    for char, digit in [
        ("\u2080", "0"),
        ("\u2081", "1"),
        ("\u2082", "2"),
        ("\u2083", "3"),
        ("\u2084", "4"),
        ("\u2085", "5"),
        ("\u2086", "6"),
        ("\u2087", "7"),
        ("\u2088", "8"),
        ("\u2089", "9"),
        ("\u208b", "-"),
    ]:
        base = base.replace(char, digit)
    base = base.lower().replace(" ", "").replace("\u2212", "-")
    # Strip trailing zeros from decimal numbers: 0.80 -> 0.8, 0.10 -> 0.1
    base = re.sub(r"(\.\d*?)0+(?=\D|$)", r"\1", base)
    return base


def extract_condition_annotation(s: str) -> str | None:
    """Extract a parenthetical condition annotation from a formula string.

    Returns the content inside trailing parentheses, if present.
    Useful for distinguishing 'Re0.77Mo0.23 (C)' from
    'Re0.77Mo0.23 (NC)' as different conditions of the same material.
    """
    match = re.search(r"\(([^)]+)\)\s*$", s.strip())
    return match.group(1).strip() if match else None


def find_best_material_match(
    query: str,
    candidates: list[str],
) -> str | None:
    """Find the best matching material name from a candidate list.

    Tries exact match, then normalized equality, then
    substring containment on normalized forms.

    Returns the matching candidate string (original form), or None.
    """
    if query in candidates:
        return query

    query_norm = normalize_formula(query)
    for candidate in candidates:
        if normalize_formula(candidate) == query_norm:
            return candidate

    # Substring containment (for cases like "x=0.23" matching "Re0.77Mo0.23")
    for candidate in candidates:
        cand_norm = normalize_formula(candidate)
        if query_norm in cand_norm or cand_norm in query_norm:
            return candidate

    return None
