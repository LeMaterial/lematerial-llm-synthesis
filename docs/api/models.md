# Data Models

All output data is represented as [Pydantic](https://docs.pydantic.dev) models.
You can serialise any model to JSON with `.model_dump()` and parse from a dict
with `.model_validate(data)`.

## Core synthesis ontology

::: llm_synthesis.models.ontologies.general.GeneralSynthesisOntology

::: llm_synthesis.models.ontologies.general.ProcessStep

::: llm_synthesis.models.ontologies.general.Material

::: llm_synthesis.models.ontologies.general.Equipment

::: llm_synthesis.models.ontologies.general.Conditions

## Paper models

::: llm_synthesis.models.paper.Paper

::: llm_synthesis.models.paper.PaperWithSynthesisOntologies

::: llm_synthesis.models.paper.SynthesisEntry

## Performance / plot models

::: llm_synthesis.models.performance.MaterialPerformanceData

::: llm_synthesis.models.performance.MaterialPlotEntry

::: llm_synthesis.models.performance.PlotMaterialMapping

::: llm_synthesis.models.performance.SeriesMapping

::: llm_synthesis.models.performance.LinkingStats
