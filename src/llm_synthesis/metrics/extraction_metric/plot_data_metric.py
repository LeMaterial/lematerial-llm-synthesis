import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import f1_score

from llm_synthesis.metrics.base import MetricInterface
from llm_synthesis.models.plot import DataSeries, ExtractedPlotData, PlotMetadata


class PlotDataMetric(MetricInterface[list[ExtractedPlotData]]):
    """
    Evaluates extracted plot data against a ground truth.
    Returns a composite score from 0.0 (worst) to 1.0 (best).
    """

    def __init__(self, weights: dict[str, float] | None = None, precision: float = 0.1):
        if weights is None:
            self.weights = {"metadata": 0.2, "series": 0.2, "numerical": 0.6}
        else:
            self.weights = weights
        self.precision = precision

    def __call__(
        self, preds: list[ExtractedPlotData], refs: list[ExtractedPlotData]
    ) -> float:
        """
        Compare predicted plot data with reference ground truth data.

        Args:
            preds: The list of predicted ExtractedPlotData objects.
            refs: The list of reference ExtractedPlotData objects.

        Returns:
            A composite score between 0.0 and 1.0.
        """
        if not preds or not refs:
            return 0.0

        # TODO: output individual scores for each plot?
        final_scores = []
        for pred_plot, ref_plot in zip(preds, refs):
            metadata_score = self._calculate_metadata_score(
                pred_plot.metadata, ref_plot.metadata
            )
            series_percentage_matched, matched_series = self._match_and_score_series(
                pred_plot.data_series, ref_plot.data_series
            )
            numerical_score = self._calculate_numerical_accuracy(
                pred_plot.data_series, ref_plot.data_series, matched_series
            )

            final_score = (
                self.weights["metadata"] * metadata_score
                + self.weights["series"] * series_percentage_matched
                + self.weights["numerical"] * numerical_score
            )
            final_scores.append(final_score)

        return np.mean(final_scores)

    def _calculate_metadata_score(
        self, pred_meta: PlotMetadata, ref_meta: PlotMetadata
    ) -> float:
        """Scores metadata based on simple string matches."""
        score = 0
        fields = ["x_axis_label", "left_y_axis_label"]
        for field in fields:
            pred_val = getattr(pred_meta, field, "").strip().lower()
            ref_val = getattr(ref_meta, field, "").strip().lower()
            if pred_val == ref_val and pred_val != "":
                score += 1
        return score / len(fields)

    def _match_and_score_series(
        self, pred_series: list[DataSeries], ref_series: list[DataSeries]
    ) -> tuple[float, dict]:
        """Matches series by name and calculates F1 score."""
        pred_names = {s.name for s in pred_series}
        ref_names = {s.name for s in ref_series}
        if not ref_names:
            return 1.0 if not pred_names else 0.0, {}
        percentage_matched = np.mean([1 if name in ref_names else 0 for name in pred_names])
        matched_series = pred_names.intersection(ref_names)
        return percentage_matched, matched_series

    def _calculate_numerical_accuracy(
        self,
        pred_series: list[DataSeries],
        ref_series: list[DataSeries],
        matched_series_names: set,
    ) -> float:
        """Calculates RMSE for matched data series using interpolation."""
        if not matched_series_names:
            return 0.0

        total_normalized_rmse = 0
        valid_series_count = 0
        
        for name in matched_series_names:
            pred_s = next(s for s in pred_series if s.name == name)
            ref_s = next(s for s in ref_series if s.name == name)

            if not pred_s.points or not ref_s.points:
                continue

            pred_pts = np.array([[p.x, p.y] for p in pred_s.points])
            ref_pts = np.array([[p.x, p.y] for p in ref_s.points])

            # Normalize points to a [0, 1] range based on the min / max of extracted and ground truth data
            ref_min = np.min(np.concatenate((pred_pts, ref_pts), axis=0), axis=0)
            ref_range = np.max(np.concatenate((pred_pts, ref_pts), axis=0), axis=0) - ref_min
            if np.any(ref_range == 0):
                continue # Avoid division by zero

            norm_pred_pts = (pred_pts - ref_min) / ref_range
            norm_ref_pts = (ref_pts - ref_min) / ref_range

            # Sort points by x-coordinate for interpolation
            pred_sorted_idx = np.argsort(norm_pred_pts[:, 0])
            ref_sorted_idx = np.argsort(norm_ref_pts[:, 0])
            
            pred_x_sorted = norm_pred_pts[pred_sorted_idx, 0]
            pred_y_sorted = norm_pred_pts[pred_sorted_idx, 1]
            ref_x_sorted = norm_ref_pts[ref_sorted_idx, 0]
            ref_y_sorted = norm_ref_pts[ref_sorted_idx, 1]

            # Create common x-grid from 0 to 1 with specified precision
            x_interp = np.arange(0, 1 + self.precision, self.precision)
            
            # Check if we have enough range for interpolation
            if (pred_x_sorted[-1] - pred_x_sorted[0] < self.precision or 
                ref_x_sorted[-1] - ref_x_sorted[0] < self.precision):
                continue
            
            # Interpolate both series on the common x-grid
            pred_y_interp = np.interp(x_interp, pred_x_sorted, pred_y_sorted)
            ref_y_interp = np.interp(x_interp, ref_x_sorted, ref_y_sorted)
            
            # Calculate RMSE between interpolated y-values
            rmse = np.sqrt(np.mean((pred_y_interp - ref_y_interp) ** 2))
            total_normalized_rmse += rmse
            valid_series_count += 1
        
        if valid_series_count == 0:
            return 0.0
            
        avg_rmse = total_normalized_rmse / valid_series_count
        
        # Convert RMSE to a score (0-1) so that lower RMSE is a higher score.
        return max(0, 1 - avg_rmse / 0.1) 