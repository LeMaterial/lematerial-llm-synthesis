"""DomainConfig: declarative specification of what a domain extracts.

A DomainConfig bundles everything the BatchRunner needs to process papers
for a given scientific domain:

  - which plots to keep (PlotFilterConfig)
  - how to prompt the material extractor
  - optional extra text-metric extraction (e.g. Tc from text)
  - optional extra VLM post-processing (e.g. geometric Tc from R(T) plots)
  - how to persist results (output writer)

Usage — built-in domains::

    from llm_synthesis.config.domain_config import DomainConfig

    config = DomainConfig.for_catalysis()
    config = (
    DomainConfig.for_superconductivity(claude_model="claude-sonnet-4-20250514")
    )
    config = DomainConfig.for_porosity()

Usage — custom domain::

    from llm_synthesis.config.domain_config import DomainConfig
    from llm_synthesis.config.plot_filter_config import PlotFilterConfig
    from llm_synthesis.runners.output_writers.json_writer import (
    AnnotatedJsonWriter
    )

    my_domain = DomainConfig(
        name="electrochemistry",
        plot_filter_config=PlotFilterConfig.for_electrochemistry(),
        material_extraction_instructions=(
            "Extract all electrode materials synthesized and tested..."
        ),
        material_output_description=(
            "Comma-separated list of electrode materials with compositions."
        ),
        text_metric_extractor=None,
        vlm_metric_processor=None,
        output_writer=AnnotatedJsonWriter(),
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field

from llm_synthesis.config.plot_filter_config import PlotFilterConfig
from llm_synthesis.domain_metrics.base import (
    BaseTextMetricExtractor,
    BaseVLMMetricProcessor,
)
from llm_synthesis.runners.output_writers.base import BaseOutputWriter


@dataclass
class DomainConfig:
    """Declarative specification of what a domain measures and how to save it.

    Attributes:
        name: Human-readable domain identifier (used in log messages).
        plot_filter_config: Determines which plots are considered relevant
            for this domain.
        material_extraction_instructions: Natural-language instructions passed
            to the material extractor LLM.  Tells it what kinds of materials
            to look for (e.g. doping variants, loading percentages, …).
        material_output_description: Describes the expected output format of
            the material extractor.
        text_metric_extractor: Optional.  Runs one extra LLM pass on the paper
            text to extract domain-specific scalars (e.g. Tc from text).
            Set to None to skip.
        vlm_metric_processor: Optional.  Runs an extra VLM pass on the
            already-filtered relevant plots to extract domain-specific
            quantities (e.g. Tc via geometric construction).
            Set to None to skip.
        output_writer: Determines how results are persisted (JSON, CSV, …).
    """

    name: str
    plot_filter_config: PlotFilterConfig
    material_extraction_instructions: str
    material_output_description: str
    output_writer: BaseOutputWriter
    text_metric_extractor: BaseTextMetricExtractor | None = field(default=None)
    vlm_metric_processor: BaseVLMMetricProcessor | None = field(default=None)

    # ------------------------------------------------------------------
    # Built-in factory methods
    # ------------------------------------------------------------------

    @classmethod
    def for_catalysis(cls) -> DomainConfig:
        """Configuration for thermocatalysis case study.

        Extracts temperature-vs-conversion/yield plots.
        No domain-specific text or VLM metric extraction beyond standard.
        """
        from llm_synthesis.runners.output_writers.json_writer import (
            AnnotatedJsonWriter,
        )

        return cls(
            name="thermocatalysis",
            plot_filter_config=PlotFilterConfig.for_catalysis(),
            material_extraction_instructions=(
                "Extract ALL distinct material compositions that were "
                "synthesized and tested in this paper. "
                "IMPORTANT: If the paper studies multiple variants of a "
                "material (e.g., different loadings, dopant concentrations, or "
                "preparation conditions), list EACH variant as a separate "
                "material. For example, if a paper studies 1%Ru/CaO, 3%Ru/CaO, "
                "and 5%Ru/CaO, list all three - do NOT merge them into a single"
                " 'Ru/CaO'. Focus on materials that were actually synthesized"
                ", not just mentioned or referenced."
            ),
            material_output_description=(
                "ALL distinct synthesized material compositions as a "
                "comma-separated list using chemical formulas. "
                "Include loading percentages and promoters when specified "
                "(e.g., '1%Ru-10%K/CaO, 3%Ru-10%K/CaO, 5%Ru-10%K/CaO, "
                "3%Ru-5%K/CaO'). Never merge variants into a single generic"
                "name."
            ),
            text_metric_extractor=None,
            vlm_metric_processor=None,
            output_writer=AnnotatedJsonWriter(),
        )

    @classmethod
    def for_superconductivity(
        cls,
        claude_model: str = "claude-sonnet-4-20250514",
    ) -> DomainConfig:
        """Configuration for the superconductors case study.

        Extracts R(T) plots and runs both text-Tc and VLM-Tc extraction.
        Results are saved as JSON per paper + a growing flat master CSV.

        Args:
            claude_model: Claude model used for VLM Tc extraction.
        """
        from llm_synthesis.domain_metrics.superconductors.tc_vlm_processor import (  # noqa: E501
            TcVLMProcessor,
        )
        from llm_synthesis.runners.output_writers.csv_writer import (
            CsvMasterWriter,
        )

        return cls(
            name="superconductors",
            plot_filter_config=PlotFilterConfig.for_superconductivity(),
            material_extraction_instructions=(
                "Extract ALL distinct superconducting material compositions "
                "that were synthesized and tested in this paper. "
                "If the paper studies multiple variants "
                "(e.g., different doping levels x=0.1, x=0.2, x=0.3), "
                "list EACH variant as a separate material."
            ),
            material_output_description=(
                "ALL distinct synthesized material compositions as a "
                "comma-separated list using chemical formulas. "
                "Include doping levels and stoichiometry."
            ),
            # text_metric_extractor is wired in BatchRunner
            # using the gemini_model
            # passed at runner construction time —
            # set to None here as a sentinel.
            text_metric_extractor=None,
            vlm_metric_processor=TcVLMProcessor(claude_model=claude_model),
            output_writer=CsvMasterWriter(master_csv_name="tc_master.csv"),
        )

    @classmethod
    def for_porosity(cls) -> DomainConfig:
        """Configuration for the porous materials / adsorption isotherm.

        Extracts pressure-vs-adsorption plots.
        No domain-specific text or VLM metric extraction beyond standard.
        """
        from llm_synthesis.runners.output_writers.json_writer import (
            AnnotatedJsonWriter,
        )

        return cls(
            name="porosity",
            plot_filter_config=PlotFilterConfig.for_coverage(),
            material_extraction_instructions=(
                "Extract ALL distinct porous or framework material compositions"
                "that were synthesized and characterized in this paper. "
                "If the paper studies multiple variants (e.g., different "
                "linkers, metal nodes, activation conditions) list EACH variant"
                " separately. Focus on materials for which adsorption or"
                " porosity data are reported."
            ),
            material_output_description=(
                "ALL distinct synthesized porous material compositions as a "
                "comma-separated list using chemical formulas or IUPAC names. "
                "Include variant labels where relevant (e.g., 'MOF-5-activated'"
                "'ZIF-8-NH2')."
            ),
            text_metric_extractor=None,
            vlm_metric_processor=None,
            output_writer=AnnotatedJsonWriter(),
        )
