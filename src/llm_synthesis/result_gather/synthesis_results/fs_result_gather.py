import base64
import json
import os
from datetime import datetime

import fsspec

from llm_synthesis.models.paper import PaperWithSynthesisOntologiesAndFigures
from llm_synthesis.result_gather.base import ResultGatherInterface


class SynthesisFSResultGather(
    ResultGatherInterface[PaperWithSynthesisOntologiesAndFigures]
):
    def __init__(self, result_dir: str = ""):
        self.result_dir = result_dir
        self.fs, _, _ = fsspec.get_fs_token_paths(self.result_dir)
        self._ensure_dir(self.result_dir)

    def gather(
        self,
        paper: PaperWithSynthesisOntologiesAndFigures,
    ):
        self._ensure_dir(os.path.join(self.result_dir, paper.id))

        # Save the main synthesis (first material's synthesis)
        with self.fs.open(
            os.path.join(self.result_dir, paper.id, "result.json"), "w"
        ) as f:
            if paper.all_syntheses:
                f.write(
                    json.dumps(
                        [
                            synthesis.model_dump()
                            for synthesis in paper.all_syntheses
                        ],
                        indent=2,
                    )
                )
            else:
                f.write(json.dumps({"error": "No synthesis found"}, indent=2))

        if paper.cost_data:
            self._save_cost_report(paper)

        with self.fs.open(
            os.path.join(self.result_dir, paper.id, "publication_text.txt"),
            "w",
        ) as f:
            f.write(paper.publication_text)

        with self.fs.open(
            os.path.join(self.result_dir, paper.id, "si_text.txt"),
            "w",
        ) as f:
            f.write(paper.si_text)

        # Save figures and extracted data in organized structure
        self._save_figures_and_data(paper)

    def _save_figures_and_data(
        self, paper: PaperWithSynthesisOntologiesAndFigures
    ):
        """Save figures and extracted plot data in an organized structure."""

        # Create figures directory
        figures_dir = os.path.join(self.result_dir, paper.id, "figures")
        self._ensure_dir(figures_dir)

        # Save each figure and its extracted data
        for i, figure in enumerate(paper.figures):
            # Get corresponding extracted data, handling potential mismatches
            extracted_data = None
            if i < len(paper.extracted_data_from_figures):
                extracted_data = paper.extracted_data_from_figures[i]
            figure_dir = os.path.join(figures_dir, f"figure_{i + 1}")
            self._ensure_dir(figure_dir)

            # Save figure image
            with self.fs.open(os.path.join(figure_dir, "image.png"), "wb") as f:
                f.write(base64.b64decode(figure.base64_data))

            # Save figure metadata
            figure_metadata = {
                "figure_id": i + 1,
                "figure_reference": figure.figure_reference,
                "figure_class": figure.figure_class,
                "quantitative": figure.quantitative,
                "alt_text": figure.alt_text,
                "position": figure.position,
                "context_before": figure.context_before,
                "context_after": figure.context_after,
            }

            with self.fs.open(
                os.path.join(figure_dir, "metadata.json"), "w"
            ) as f:
                f.write(json.dumps(figure_metadata, indent=2))

            # Save extracted plot data
            if extracted_data is not None:
                # Try to save as structured JSON if it's a Pydantic model
                try:
                    if hasattr(extracted_data, "model_dump"):
                        plot_data_json = extracted_data.model_dump()
                    else:
                        # Fallback to string representation
                        plot_data_json = {"raw_data": str(extracted_data)}

                    with self.fs.open(
                        os.path.join(figure_dir, "extracted_data.json"), "w"
                    ) as f:
                        f.write(json.dumps(plot_data_json, indent=2))

                except Exception:
                    # Fallback to text file
                    with self.fs.open(
                        os.path.join(figure_dir, "extracted_data.txt"), "w"
                    ) as f:
                        f.write(str(extracted_data))
            else:
                # Save empty data indicator
                with self.fs.open(
                    os.path.join(figure_dir, "extracted_data.json"), "w"
                ) as f:
                    f.write(
                        json.dumps({"error": "No data extracted"}, indent=2)
                    )

        # Save summary of all figures
        figures_summary = {
            "total_figures": len(paper.figures),
            "quantitative_figures": len(
                [f for f in paper.figures if f.quantitative]
            ),
            "figure_types": {},
            "successful_extractions": len(
                [d for d in paper.extracted_data_from_figures if d is not None]
            ),
            "figures": [
                {
                    "id": i + 1,
                    "figure_class": figure.figure_class,
                    "quantitative": figure.quantitative,
                    "figure_reference": figure.figure_reference,
                    "data_extracted": i < len(paper.extracted_data_from_figures)
                    and paper.extracted_data_from_figures[i] is not None,
                }
                for i, figure in enumerate(paper.figures)
            ],
        }

        # Count figure types
        for figure in paper.figures:
            fig_type = figure.figure_class
            figures_summary["figure_types"][fig_type] = (
                figures_summary["figure_types"].get(fig_type, 0) + 1
            )

        with self.fs.open(os.path.join(figures_dir, "summary.json"), "w") as f:
            f.write(json.dumps(figures_summary, indent=2))

    def _save_cost_report(self, paper: PaperWithSynthesisOntologiesAndFigures):
        """Save cost information to JSON format."""

        # Save detailed cost report as JSON
        cost_report = {
            "timestamp": datetime.now().isoformat(),
            "paper_id": paper.id,
            "cost_breakdown_usd": paper.cost_data.get("breakdown", {}),
            "total_cost_usd": paper.cost_data.get("total_cost", 0.0),
            "model_info": paper.cost_data.get("models", {}),
            "statistics": {
                "total_llm_calls": paper.cost_data.get("total_calls", 0),
                "materials_processed": paper.cost_data.get(
                    "materials_count", 0
                ),
                "synthesis_extractions": paper.cost_data.get(
                    "synthesis_calls", 0
                ),
                "material_extractions": paper.cost_data.get(
                    "material_calls", 0
                ),
                "judge_evaluations": paper.cost_data.get("judge_calls", 0),
            },
        }

        with self.fs.open(
            os.path.join(self.result_dir, paper.id, "cost_report.json"), "w"
        ) as f:
            f.write(json.dumps(cost_report, indent=2))

    def _ensure_dir(self, dir: str):
        if not self.fs.exists(dir):
            self.fs.makedirs(dir)
