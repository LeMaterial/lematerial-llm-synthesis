import logging

from llm_synthesis.metrics.extraction_metric.base import (
    LinePlotExtractionMetric,
)
from llm_synthesis.models.plot import ExtractedLinePlotData

logger = logging.getLogger(__name__)


class FigureExtractionMetric(LinePlotExtractionMetric):
    def __call__(
        self, preds: ExtractedLinePlotData, refs: ExtractedLinePlotData
    ) -> float:
        """
        Compute average RMSE across all matching series.
        For each data series, for each data point in the extracted data, find
        the closest point in the ground truth data and compute the squared distance.
        Then take the square root of the average squared distance.

        Returns:
            float: Average RMSE across all series.
        """
        extracted_name_to_coordinates = preds.name_to_coordinates
        gt_name_to_coordinates = refs.name_to_coordinates

        extracted_keys = set(extracted_name_to_coordinates)
        gt_keys = set(gt_name_to_coordinates)

        missing_keys = gt_keys - extracted_keys
        if missing_keys:
            logging.info(f"Series missing in LLM output: {missing_keys}.")

        common_keys = extracted_keys & gt_keys
        x_scale, y_scale = self.compute_scale(gt_name_to_coordinates)

        rmse_list = [
            self.pointwise_rmse(
                extracted_name_to_coordinates[k],
                gt_name_to_coordinates[k],
                x_scale,
                y_scale,
            )
            for k in common_keys
        ]

        return sum(rmse_list) / len(rmse_list)

    @staticmethod
    def compute_scale(
        ground_truth: dict[str, list[tuple[float, float]]],
    ) -> tuple[float, float]:
        """
        Use the difference between the max and min values of x and y coordinates
        across all series to compute the scale for normalization.

        Args:
            ground_truth (dict): Dictionary mapping series names to their 2D-coordinates.
        Returns:
            tuple: (x_scale, y_scale) wthe ranges of x and y coordinates.
        """
        all_x = [x for coords in ground_truth.values() for x, _ in coords]
        all_y = [y for coords in ground_truth.values() for _, y in coords]
        x_scale = max(all_x) - min(all_x) or 1e-8
        y_scale = max(all_y) - min(all_y) or 1e-8
        return x_scale, y_scale

    @staticmethod
    def pointwise_rmse(
        extracted_coords: list[tuple[float, float]],
        gt_coords: list[tuple[float, float]],
        x_scale: float,
        y_scale: float,
    ) -> float:
        """Compute RMSE for one series using nearest-neighbor matching."""
        if not extracted_coords:
            return 0.0

        total_sq_error = 0.0
        for llm_x, llm_y in extracted_coords:
            min_dist_sq = min(
                ((gt_x - llm_x) / x_scale) ** 2
                + ((gt_y - llm_y) / y_scale) ** 2
                for gt_x, gt_y in gt_coords
            )
            total_sq_error += min_dist_sq

        return (total_sq_error / len(extracted_coords)) ** 0.5
