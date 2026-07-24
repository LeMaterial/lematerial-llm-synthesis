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


# Element symbols, used only to tokenize a formula into (element,
# coefficient) pairs without mistaking a doping variable for part of a
# two-letter symbol (e.g. "Fx" must split as F + x, not read as one token).
_ELEMENTS = {
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
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
    "Y",
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
    "I",
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
    "W",
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
    "Ac",
    "Th",
    "Pa",
    "U",
    "Np",
    "Pu",
}


def _tokenize_elements_with_coefficients(s: str) -> list[tuple[str, str]]:
    """Split a formula into (element_symbol, coefficient_str) pairs, in
    order, using the known element-symbol table (not a greedy regex) so a
    doping variable right after an element isn't mistaken for part of a
    two-letter symbol -- e.g. "Fx" splits as ("F", "x"), not one token.

    coefficient_str is the raw run of digits/./-/+/x/y/z following the
    symbol (e.g. "1-x", "0.9", "x", or "" if the element has no explicit
    coefficient, i.e. an implicit 1).
    """
    tokens: list[tuple[str, str]] = []
    i, n = 0, len(s)
    while i < n:
        ch = s[i]
        if ch.isalpha() and ch.isupper():
            two = s[i : i + 2]
            if len(two) == 2 and two[1].islower() and two in _ELEMENTS:
                sym = two
            elif ch in _ELEMENTS:
                sym = ch
            else:
                i += 1
                continue
            i += len(sym)
            coef_start = i
            while i < n and (
                s[i].isdigit() or s[i] in ".-+" or s[i].lower() in "xyz"
            ):
                # Don't swallow the start of the next element (e.g. the
                # "F" in a following "FeAs") if it happens to be an
                # uppercase X/Y that also starts a real 2-letter symbol.
                if s[i].isupper() and s[i : i + 2] in _ELEMENTS:
                    break
                i += 1
            tokens.append((sym, s[coef_start:i]))
        else:
            i += 1
    return tokens


def is_doping_instance(general: str, specific: str) -> bool:
    """True if `specific` is a concrete doping level of the parameterized
    formula `general` -- e.g. general="CaFe1-xCoxAsF",
    specific="CaFe0.9Co0.1AsF" -- and NOT true merely because the two
    formulas share some elements or a similar length.

    Requires, element-position by element-position:
    - the same elements in the same order (different dopant or different
      substituent element -> reject, e.g. "CaFe1-xCoxAsF" vs
      "CaFe1-xNixAsF" do NOT match each other).
    - at any position where `general`'s coefficient contains a doping
      variable (x, y, or z), `specific`'s coefficient there may be any
      concrete number.
    - at every OTHER position, `specific`'s coefficient must match
      `general`'s exactly (numerically) -- a fixed stoichiometric
      coefficient is not allowed to silently change. This is what
      rejects a case like general="CaxO3" (only Ca is doped) vs
      specific="Ca0.4O7" (O's coefficient changed from a fixed 3 to 7,
      which the doping variable at Ca's position does not license).
    - at least one position has a doping variable at all -- two already-
      concrete formulas that happen to share an element skeleton (e.g. a
      condition variant like "Re0.77Mo0.23 (NC)" vs "Re0.77Mo0.23") are a
      duplicate/near-duplicate, not a doping-instance relationship.

    This is intentionally conservative: a false positive here would
    silently invent a material entry from an LLM/VLM-proposed name that
    doesn't actually appear in the paper's material list.
    """
    gen_toks = _tokenize_elements_with_coefficients(general)
    spec_toks = _tokenize_elements_with_coefficients(specific)
    if [e for e, _ in gen_toks] != [e for e, _ in spec_toks]:
        return False

    has_variable = False
    for (_, gen_coef), (_, spec_coef) in zip(gen_toks, spec_toks):
        if re.search(r"[xyz]", gen_coef, re.IGNORECASE):
            has_variable = True
            # specific's coefficient at a doped position must be a
            # concrete, non-empty number -- an empty/unparseable
            # coefficient here means specific never actually instantiated
            # this position (e.g. a series label that just repeats the
            # "x=..." annotation as loose text instead of substituting a
            # real number into the formula), so it is not yet a genuine
            # doping instance.
            if not re.match(r"^\d+\.?\d*$", spec_coef):
                return False
            continue
        # Fixed (non-doped) position: coefficients must match exactly.
        gen_norm = (
            gen_coef.rstrip("0").rstrip(".") if "." in gen_coef else gen_coef
        )
        spec_norm = (
            spec_coef.rstrip("0").rstrip(".") if "." in spec_coef else spec_coef
        )
        if gen_norm != spec_norm:
            return False

    return has_variable
