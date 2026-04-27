# Multi-LLM Synthesis Extraction and Evaluation



Configure llm_names list in multi_llm.yaml configs in configs/synthesis_extraction, and configs/judge directories to run mxn synthesis extractions/judge evaluations. (For now material extraction also runs with the same llms' configured in synthesis extraction config)

## Command to run

```
uv run examples/scripts/deployment/extract_synthesis_multi_llm_judge.py \
  data_loader=local \
  data_loader.architecture.data_dir="/path/to/markdown" \
  synthesis_extraction=multi_llm \
  material_extraction=multi_llm \
  judge=multi_llm \
  result_save=multi_llm
```

\
## Output

- `result.json` - Detailed results of mxn evaluations
- `evaluation_matrix.png` - Per-paper mxn evaluation scores
- `global_avg_evaluation_matrix.png` - Average scores across all papers
- `cost_report.json` - Cost breakdown

Results saved to `results/single_run/<timestamp>/`
