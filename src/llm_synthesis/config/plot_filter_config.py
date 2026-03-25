"""Configuration for plot filtering in performance linking.

This module provides configurable filtering criteria for determining which plots
are relevant for performance data extraction, allowing domain-specific 
customization.

Example usage:
    # Default catalysis configuration
    config = PlotFilterConfig()

    # Custom configuration for electrochemistry
    config = PlotFilterConfig(
        x_axis_labels=["potential", "voltage", "e"],
        x_axis_units=["v", "mv"],
        y_axis_keywords=["current", "capacitance"],
    )
"""

from pydantic import BaseModel, Field


class PlotFilterConfig(BaseModel):
    """Configuration for filtering plots based on axis characteristics.

    This allows domain-specific customization of what constitutes a
    "relevant" plot for performance data extraction.

    Attributes:
        x_axis_labels: Labels that indicate a relevant x-axis (case-insensitive)
        x_axis_units: Units that indicate a relevant x-axis (case-insensitive)
        y_axis_keywords: Keywords in y-axis label indicating performance metrics
        y_axis_units: Units that suggest performance data (e.g., "%")
        require_y_keyword_with_percentage: If True, % unit alone is not enough;
            the label must also contain a y_axis_keyword
        filter_x_axis: Whether to apply x-axis filtering
        filter_y_axis: Whether to apply y-axis filtering
    """

    # X-axis configuration (default: temperature for catalysis)
    x_axis_labels: list[str] = Field(
        default=["temperature", "temp"],
        description=(
            "X-axis labels that indicate relevance "
            "(substring match, case-insensitive)"
        ),
    )
    x_axis_units: list[str] = Field(
        default=[
            "°c",
            "°k",
            "°f",
            "ºc",
            "ºk",
            "k",
            "c",
            "f",
            "kelvin",
            "celsius",
        ],
        description="X-axis units that indicate relevance (case-insensitive)",
    )

    # Y-axis configuration (default: conversion/performance metrics)
    y_axis_keywords: list[str] = Field(
        default=["conversion", "yield", "activity"],
        description="Keywords in y-axis label indicating performance metrics",
    )
    y_axis_units: list[str] = Field(
        default=["%", "percent"],
        description="Y-axis units that suggest performance data",
    )

    # X-axis symbol detection (e.g., "X" for conversion in catalysis)
    x_symbol_prefixes: list[str] = Field(
        default=["x_", "x "],
        description="Prefixes for conversion symbol (e.g., 'X_NH3')",
    )
    x_symbol_exact: list[str] = Field(
        default=["x"],
        description="Exact matches for conversion symbol",
    )

    # Exclusion patterns for y-axis (reject if label matches any of these)
    y_axis_exclude_patterns: list[str] = Field(
        default=[],
        description=(
            "Patterns in y-axis label that indicate a derived/non-raw plot "
            "even if a y_axis_keyword matches. Checked as substrings after "
            "lowercasing. E.g., 'ρ-ρ' excludes difference-resistivity plots."
        ),
    )

    # Filtering behavior
    require_y_keyword_with_percentage: bool = Field(
        default=True,
        description=(
            "If True, % unit alone is not enough; "
            "the label must also contain a y_axis_keyword. "
            "This prevents matching characterization plots like TPR/TPD."
        ),
    )
    filter_x_axis: bool = Field(
        default=True,
        description="Whether to apply x-axis filtering",
    )
    filter_y_axis: bool = Field(
        default=True,
        description="Whether to apply y-axis filtering",
    )

    @staticmethod
    def _normalize_axis_text(text: str) -> str:
        """Normalize axis label/unit text for matching.

        Strips LaTeX formatting and converts LaTeX symbols to Unicode 
        equivalents so that e.g. '$\\rho_{xx}$' matches keyword 'ρ'.
        """
        # Strip LaTeX dollar signs
        text = text.replace("$", "")
        # Convert common LaTeX symbols to Unicode
        # Use raw strings to match literal backslash sequences
        latex_to_unicode = [
            ("\\rho", "ρ"),
            ("\\omega", "ω"),
            ("\\mu", "μ"),
            ("\\delta", "δ"),
            ("\\sigma", "σ"),
            ("\\alpha", "α"),
            ("\\beta", "β"),
            ("\\gamma", "γ"),
            ("\\chi", "χ"),
            ("\\lambda", "λ"),
            ("\\cdot", "·"),
        ]
        for latex, uni in latex_to_unicode:
            text = text.replace(latex, uni)
        # Also handle case where \r was interpreted as carriage return
        # (happens when Python string literal has \rho without raw prefix)
        text = text.replace("\rho", "ρ")
        text = text.replace("\mu", "μ")
        return text

    def is_relevant_x_axis(self, label: str | None, unit: str | None) -> bool:
        """Check if x-axis indicates a relevant plot.

        Args:
            label: X-axis label (e.g., "Temperature")
            unit: X-axis unit (e.g., "°C")

        Returns:
            True if x-axis matches configured criteria
        """
        if not self.filter_x_axis:
            return True

        label_lower = self._normalize_axis_text((label or "").lower().strip())
        unit_lower = self._normalize_axis_text((unit or "").lower().strip())

        # Check if label contains any configured labels (substring match)
        if any(t in label_lower for t in self.x_axis_labels):
            return True

        # Check if unit matches any configured units (exact match for units)
        if any(u == unit_lower for u in self.x_axis_units):
            return True

        return False

    def is_relevant_y_axis(self, label: str | None, unit: str | None) -> bool:
        """Check if y-axis indicates a performance metric.

        Args:
            label: Y-axis label (e.g., "Conversion")
            unit: Y-axis unit (e.g., "%")

        Returns:
            True if y-axis matches configured criteria
        """
        if not self.filter_y_axis:
            return True

        label_lower = self._normalize_axis_text((label or "").lower().strip())
        unit_lower = self._normalize_axis_text((unit or "").lower().strip())

        # If y-axis is completely empty, we can't verify — reject it
        if not label_lower and not unit_lower:
            return False

        # Check exclusion patterns first — reject derived/difference plots
        if any(pat in label_lower for pat in self.y_axis_exclude_patterns):
            return False

        # Check for conversion keywords in label
        has_keyword = any(kw in label_lower for kw in self.y_axis_keywords)

        # Check for conversion symbol (e.g., "X" for conversion)
        is_x_symbol = label_lower in self.x_symbol_exact or any(
            label_lower.startswith(p) for p in self.x_symbol_prefixes
        )

        # Check for percentage unit
        has_percentage_unit = unit_lower in self.y_axis_units

        # Main logic
        if has_percentage_unit:
            if self.require_y_keyword_with_percentage:
                # % unit requires keyword/symbol confirmation
                return has_keyword or is_x_symbol
            else:
                # % unit alone is sufficient
                return True

        # Keywords/symbols alone are sufficient
        return has_keyword or is_x_symbol

    @classmethod
    def for_catalysis(cls) -> "PlotFilterConfig":
        """Factory method for catalysis domain (default configuration)."""
        return cls()

    @classmethod
    def for_electrochemistry(cls) -> "PlotFilterConfig":
        """Factory method for electrochemistry domain."""
        return cls(
            x_axis_labels=["potential", "voltage", "e", "v"],
            x_axis_units=["v", "mv", "v vs. rhe", "v vs rhe"],
            y_axis_keywords=["current", "capacitance", "capacity", "coulombic"],
            y_axis_units=["%", "percent", "ma", "a", "f/g", "mah/g"],
            require_y_keyword_with_percentage=False,
        )

    @classmethod
    def for_superconductivity(cls) -> "PlotFilterConfig":
        """Factory method for superconductivity domain (R(T) plots)."""
        return cls(
            x_axis_labels=[
                "temperature", "temp", "t (k)", "t(k)", "t [k]", "t[k]"
            ],
            x_axis_units=["k", "°k", "kelvin"],
            y_axis_keywords=[
                "resistance", "resistivity", "r(t)", "r/r",
                "ρ", "rho", "normalized resistance", "r (ω",
                "r (m", "r (μ", "r [ω", "r [m", "r [μ",
                "ρ (", "ρ [", "ρ/ρ",
            ],
            y_axis_units=[
                "ω", "ohm", "mω", "μω", "ω·cm", "μω·cm", "mω·cm",
                "ω cm", "μω cm", "mω cm", "ωcm", "μωcm", "mωcm",
                "ω⋅cm", "μω⋅cm", "mω⋅cm",
                "a.u.",
            ],
            y_axis_exclude_patterns=[
                # Difference / subtracted quantities
                "ρ-ρ", "r-r", "ρ−ρ", "r−r",  # minus sign variants
                "ρ - ρ", "r - r",              # spaced minus
                "δρ", "δr", "Δρ", "Δr",        # delta variants
                # Derivatives
                "dρ/dt", "dr/dt", "dρ/d", "dr/d",
                # Ratio to residual resistivity (but NOT normalized to room 
                # temp)
                # "ρ/ρ₀", "r/r₀", "r/r0" are residual-ratio plots (not useful)
                # "ρ/ρ₃₀₀" or "r/r(300)" are room-temp-normalized R(T) (useful!)
                "ρ/ρ₀", "ρ/ρ0",
                "r/r₀", "r/r0",
            ],
            require_y_keyword_with_percentage=False,
        )

    @classmethod
    def no_filter(cls) -> "PlotFilterConfig":
        """Factory method that disables all filtering (link all plots)."""
        return cls(
            filter_x_axis=False,
            filter_y_axis=False,
        )
