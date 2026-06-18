# Transformers

All extractors inherit from `ExtractorInterface[T, R]`. Each implements a `forward(input)`
method (synchronous) and gets an `aforward(input)` method (async) for free.

## Base interface

::: llm_synthesis.transformers.base.ExtractorInterface

## Material extraction

::: llm_synthesis.transformers.material_extraction.dspy_extraction.DspyTextExtractor

::: llm_synthesis.transformers.material_extraction.dspy_extraction.make_dspy_text_extractor_signature

## Synthesis extraction

::: llm_synthesis.transformers.synthesis_extraction.dspy_synthesis_extraction.DspySynthesisExtractor

::: llm_synthesis.transformers.synthesis_extraction.dspy_synthesis_extraction.make_dspy_synthesis_extractor_signature

## PDF extraction

::: llm_synthesis.transformers.pdf_extraction.docling_pdf_extractor.DoclingPDFExtractor

::: llm_synthesis.transformers.pdf_extraction.mistral_pdf_extractor.MistralPDFExtractor

## Plot data extraction

::: llm_synthesis.transformers.plot_extraction.claude_extraction.plot_data_extraction.ClaudeLinePlotDataExtractor

::: llm_synthesis.transformers.plot_extraction.litellm_plot_data_extraction.LiteLLMPlotDataExtractor

## Performance linking

::: llm_synthesis.transformers.performance_linking.series_material_linker.SeriesMaterialLinker
