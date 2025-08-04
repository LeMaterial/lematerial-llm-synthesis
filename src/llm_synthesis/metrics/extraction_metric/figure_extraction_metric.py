import logging
from typing import Literal

from llm_synthesis.metrics.extraction_metric.base import (
    LinePlotExtractionMetric,
)
from llm_synthesis.models.plot import ExtractedLinePlotData


class FigureExtractionMetric(LinePlotExtractionMetric):
    def __call__(
        self,
        preds: ExtractedLinePlotData,
        refs: ExtractedLinePlotData,
        error_metric: Literal["rmse", "mae"] = "rmse",
    ) -> float:
        """
        Compute average RMSE or MAE across all matching series.
        """
        extracted = preds.name_to_coordinates
        ground_truth = refs.name_to_coordinates

        missing_keys = set(ground_truth) - set(extracted)
        if missing_keys:
            logging.info(f"Series missing in LLM output: {missing_keys}.")

        common_keys = set(extracted) & set(ground_truth)
        if not common_keys:
            logging.warning(
                "No common series names found between ground truth and LLM output."
            )
            return None

        x_scale, y_scale = self.compute_scale(ground_truth)

        error_function = (
            self.pointwise_rmse
            if error_metric == "rmse"
            else self.pointwise_mae
        )

        errors = [
            error_function(extracted[k], ground_truth[k], x_scale, y_scale)
            for k in common_keys
        ]

        return sum(errors) / len(errors)

    @staticmethod
    def compute_scale(
        ground_truth: dict[str, list[tuple[float, float]]],
    ) -> tuple[float, float]:
        """Compute normalization scales for x and y."""
        all_x = [x for coords in ground_truth.values() for x, _ in coords]
        all_y = [y for coords in ground_truth.values() for _, y in coords]
        x_range = max(all_x) - min(all_x) or 1e-8
        y_range = max(all_y) - min(all_y) or 1e-8
        return x_range, y_range

    @staticmethod
    def pointwise_rmse(
        extracted_coords: list[tuple[float, float]],
        gt_coords: list[tuple[float, float]],
        x_scale: float,
        y_scale: float,
    ) -> float:
        """Compute RMSE using nearest-neighbor matching for one series."""
        if not extracted_coords:
            return 0.0

        total_sq_error = sum(
            min(
                ((gt_x - ex_x) / x_scale) ** 2 + ((gt_y - ex_y) / y_scale) ** 2
                for gt_x, gt_y in gt_coords
            )
            for ex_x, ex_y in extracted_coords
        )

        return (total_sq_error / len(extracted_coords)) ** 0.5

    @staticmethod
    def pointwise_mae(
        extracted_coords: list[tuple[float, float]],
        gt_coords: list[tuple[float, float]],
        x_scale: float,
        y_scale: float,
    ) -> float:
        """Compute MAE using nearest-neighbor matching for one series."""
        if not extracted_coords:
            return 0.0

        total_abs_error = sum(
            min(
                (
                    ((ex_x - gt_x) / x_scale) ** 2
                    + ((ex_y - gt_y) / y_scale) ** 2
                )
                ** 0.5
                for gt_x, gt_y in gt_coords
            )
            for ex_x, ex_y in extracted_coords
        )

        return total_abs_error / len(extracted_coords)
