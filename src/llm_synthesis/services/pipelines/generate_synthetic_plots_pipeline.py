import json
import os
import random
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import skewnorm

from llm_synthesis.services.pipelines.base_pipeline import BasePipeline
from llm_synthesis.utils.synthetic_figure_utils import (
    colors,
    filler_axis_labels,
    filler_list,
    matrix_list,
    positions,
    shapes,
    x_axis_labels,
    y_axis_labels,
)


class GenerateSyntheticPlotsPipeline(BasePipeline):
    def __init__(
        self,
        num_plots: int,
        images_path: str,
        groundtruths_path: str,
        seed: int = 42,
    ):
        self.num_plots = num_plots
        self.images_path = images_path
        self.groundtruths_path = groundtruths_path
        self.seed = seed

    def run(self) -> None:
        """
        Generate synthetic scatter plots and save them
        along with their ground truth coordinates.

        Args:
            num_plots (int): Number of plots to generate.
            images_path (str): Directory to save the generated images.
            groundtruths_path (str): Directory to save the ground truth
            coordinates under the format {name_of_group:
            [[x1, y1], [x2, y2], ...]}.
            seed (int): Random seed for reproducibility.
        """
        for i in range(self.num_plots):
            random.seed(self.seed + i)
            np.random.seed(i)
            image_name = f"figure_{i}.png"
            image_path = os.path.join(self.images_path, image_name)
            groundtruth_path = os.path.join(
                self.groundtruths_path, image_name.replace("png", "json")
            )
            self._plot_multiple_subplots(image_path, groundtruth_path)

    @staticmethod
    def generate_random_data(
        x: np.ndarray, curve_type: str
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate random y values based on the specified curve type.

        Args:
            x (np.ndarray): The x values for the scatter plot.
            curve_type (str): The type of curve to generate.
        """
        num_points = len(x)
        if curve_type == "exp_increasing":
            y = np.exp(0.3 * x) + np.random.normal(
                0, 0.2 * np.exp(0.4 * x), num_points
            )
        elif curve_type == "exp_decreasing":
            y = np.exp(-0.3 * x) + np.random.normal(
                0, 0.2 * np.exp(-0.4 * x), num_points
            )
        elif curve_type == "exp_increasing_dec_rate":
            y = np.exp(0.1 * x) + np.random.normal(
                0, 0.2 * np.exp(0.2 * x), num_points
            )
        elif curve_type == "exp_decreasing_inc_rate":
            y = np.exp(-0.1 * x) + np.random.normal(
                0, 0.2 * np.exp(-0.2 * x), num_points
            )
        elif curve_type == "linear_steep":
            y = 3 * x + np.random.normal(0, 5, num_points)
        elif curve_type == "linear_shallow":
            y = 0.5 * x + np.random.normal(0, 2, num_points)
        else:
            raise ValueError(f"Unknown curve_type: {curve_type}")

        y += np.random.uniform(-8, 8)
        return x, y

    @staticmethod
    def _random_x_axis(ltype: float) -> str:
        return random.choice(
            filler_axis_labels if ltype >= 0.4 else x_axis_labels
        )

    @staticmethod
    def _random_filler() -> str:
        return random.choice(filler_list)

    @staticmethod
    def _random_matrix() -> str:
        return random.choice(matrix_list)

    @staticmethod
    def _skewed_normal_distribution(
        num_groups: int, center: int = 6, skew: int = -1, size: int = 1
    ) -> int:
        """
        Generate a skewed normal distribution for
        the number of points in each group.

        Args:
            num_groups (int): Number of groups to generate.
            center (int): Center of the distribution.
            skew (int): Skewness of the distribution.
            size (int): Number of samples to generate.
        Returns:
            int: Number of points in each group, clipped to a range of 3 to
        """
        if num_groups <= 2:
            center += 2
        return int(
            np.clip(
                skewnorm.rvs(a=skew, loc=center, scale=2, size=size), 3, 20
            )[0]
        )

    @staticmethod
    def _skewed_marker_size(
        center: int = 50, skew: int = 4, size: int = 1
    ) -> float:
        """Generate a skewed normal distribution for marker sizes.
        Args:
            center (int): Center of the distribution.
            skew (int): Skewness of the distribution.
            size (int): Number of samples to generate.
        Returns:
            float: Marker size, clipped to a range of 10 to 200.
        """
        return float(
            np.clip(
                skewnorm.rvs(a=skew, loc=center, scale=20, size=size), 10, 200
            )[0]
        )

    @staticmethod
    def _generate_x_points(num_points: int) -> np.ndarray:
        """Generate x points for the scatter plot.
        Args:
            num_points (int): Number of points to generate.

        Returns:
            np.ndarray: Sorted x points, either uniformly
            distributed or randomly selected.
        """
        return (
            np.sort(np.random.choice(range(0, 20), num_points, replace=False))
            if random.random() < 0.5
            else np.sort(np.random.uniform(0, 20, num_points))
        )

    @staticmethod
    def _pick_color(used_colors: set[str]) -> str:
        available_colors = set(colors) - used_colors
        color = random.choice(list(available_colors))
        return color

    @staticmethod
    def _pick_shape(used_shapes: set[str]) -> str:
        available_shapes = set(shapes) - used_shapes
        shape = random.choice(list(available_shapes))
        return shape

    @staticmethod
    def _generate_legend_label(
        legend_type: float, default_fill: str, default_mat: str
    ) -> str | None:
        if legend_type < 0.2:
            return f"{round(random.random(), 2)}% {default_fill}/{default_mat}"
        elif legend_type < 0.4:
            return f"{round(random.random(), 2)}% {default_fill}"
        elif legend_type < 0.6:
            return f"{GenerateSyntheticPlotsPipeline._random_filler()}/{default_mat}"  # noqa: E501
        elif legend_type < 0.8:
            return GenerateSyntheticPlotsPipeline._random_matrix()
        return None

    @staticmethod
    def _adjust_y_axis(ax: plt.Axes) -> None:
        space_top = random.random() < 0.5
        ylim = ax.get_ylim()
        ylim_range = ylim[1] - ylim[0]
        ax.set_ylim(
            ylim[0], ylim[1] + 0.2 * ylim_range
        ) if space_top else ax.set_ylim(ylim[0] - 0.2 * ylim_range, ylim[1])

    @staticmethod
    def _place_legend(
        ax: plt.Axes, legend_handles: list[Any], legend_labels: list[str]
    ) -> None:
        """
        Place the legend in the best position to avoid
        overlap with scatter points. It iterates through predefined
        positions and selects the one with the least overlap.
        If no position is suitable, it defaults to the upper right corner.
        """
        best_pos = "upper right"
        min_overlap = float("inf")
        for pos in positions:
            legend = ax.legend(
                handles=legend_handles,
                labels=legend_labels,
                loc=pos,
                frameon=False,
            )
            bbox = legend.get_window_extent().transformed(
                ax.transData.inverted()
            )
            overlap = sum(
                1
                for scatter in ax.collections
                for x, y in scatter.get_offsets()
                if bbox.x0 <= x <= bbox.x1 and bbox.y0 <= y <= bbox.y1
            )
            if overlap < min_overlap:
                min_overlap = overlap
                best_pos = pos
            legend.remove()
        ax.legend(
            handles=legend_handles,
            labels=legend_labels,
            loc=best_pos,
            frameon=False,
        )

    def _plot_random_scatter_on_ax(
        self, ax: plt.Axes
    ) -> tuple[dict[str, list[list[float]]], str, str]:
        """
        Plot a random scatter plot on the given Axes object.
        Returns:
            A tuple containing:
            - A dictionary mapping legend labels to their 2D coordinates
            - The x-axis label
            - The y-axis label
        """
        num_groups = random.randint(1, 4)
        with_lines = random.random() < 0.5

        num_points = max(
            3,
            min(
                20,
                GenerateSyntheticPlotsPipeline._skewed_normal_distribution(
                    num_groups
                ),
            ),
        )

        # x-axis consistency controls whether all groups share the same x points
        # or each group has its own randomly generated x points
        consistent_x = random.random() < 0.75
        x = (
            GenerateSyntheticPlotsPipeline._generate_x_points(num_points)
            if consistent_x
            else None
        )

        legend_type = random.random()
        marker_size = GenerateSyntheticPlotsPipeline._skewed_marker_size()
        default_mat = GenerateSyntheticPlotsPipeline._random_matrix()
        default_fill = GenerateSyntheticPlotsPipeline._random_filler()
        x_label = GenerateSyntheticPlotsPipeline._random_x_axis(legend_type)
        y_label = random.choice(y_axis_labels)

        curve_types = random.choice(
            [
                [
                    "exp_increasing",
                    "exp_decreasing",
                    "exp_increasing_dec_rate",
                    "exp_decreasing_inc_rate",
                ],
                ["linear_steep", "linear_shallow"],
            ]
        )
        line_type = (random.choice(["best_fit", "connecting_lines"]) if with_lines else None)

        used_colors = set()
        used_shapes = set()
        legend_handles = []
        legend_labels = []
        label_to_coordinates = {}        

        for _ in range(num_groups):
            group_x = (
                x
                if consistent_x
                else GenerateSyntheticPlotsPipeline._generate_x_points(
                    num_points
                )
            )
            group_x, group_y = (
                GenerateSyntheticPlotsPipeline.generate_random_data(
                    group_x, random.choice(curve_types)
                )
            )

            color = GenerateSyntheticPlotsPipeline._pick_color(used_colors)
            shape = GenerateSyntheticPlotsPipeline._pick_shape(used_shapes)
            used_colors.add(color)
            used_shapes.add(shape)
            edge_flag = shape in ["o", "s"] and random.random() < 0.33

            scatter = ax.scatter(
                group_x,
                group_y,
                edgecolor=color if edge_flag else None,
                facecolor="none" if edge_flag else None,
                c=color if not edge_flag else None,
                marker=shape,
                s=marker_size,
            )
            legend_handles.append(scatter)

            if line_type == "best_fit":
                m, b = np.polyfit(group_x, group_y, 1)
                ax.plot(group_x, m * group_x + b, c=color)
            elif line_type == "connecting":
                ax.plot(group_x, group_y, c=color)

            label = GenerateSyntheticPlotsPipeline._generate_legend_label(
                legend_type, default_fill, default_mat
            )
            if label:
                legend_labels.append(label)

            label_to_coordinates[label] = [
                [float(xi), float(yi)] for xi, yi in zip(group_x, group_y)
            ]

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        GenerateSyntheticPlotsPipeline._adjust_y_axis(ax)

        if legend_type < 0.8:
            GenerateSyntheticPlotsPipeline._place_legend(
                ax, legend_handles, legend_labels
            )

        return label_to_coordinates, x_label, y_label

    def _plot_random_boxplot_on_ax(
            self, ax: plt.Axes
        ) -> tuple[dict[str, list[list[float]]], str, str]:
            """Plot a random box plot on the provided Axes."""
            num_groups = random.randint(1, 4)
            label_to_coordinates: dict[str, list[list[float]]] = {}
            data = []
            labels = []
            used_colors: set[str] = set()

            for group_idx in range(num_groups):
                group_size = GenerateSyntheticPlotsPipeline._skewed_normal_distribution(
                    num_groups
                )
                group_data = np.random.normal(
                    loc=random.uniform(-5, 5),
                    scale=random.uniform(0.5, 3.0),
                    size=group_size,
                )
                label = GenerateSyntheticPlotsPipeline._random_filler()
                labels.append(label)
                data.append(group_data)
                label_to_coordinates[label] = [
                    [float(group_idx + 1), float(val)] for val in group_data
                ]

            box = ax.boxplot(data, patch_artist=True)
            for patch in box["boxes"]:
                color = GenerateSyntheticPlotsPipeline._pick_color(used_colors)
                used_colors.add(color)
                patch.set_facecolor(color)

            x_label = GenerateSyntheticPlotsPipeline._random_x_axis(0.5)
            y_label = random.choice(y_axis_labels)
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.set_xticklabels(labels)
            GenerateSyntheticPlotsPipeline._adjust_y_axis(ax)
            return label_to_coordinates, x_label, y_label
    
    def _plot_multiple_subplots(
        self, image_path: str, groundtruth_path: str
    ) -> None:
        # randomize the number of rows and columns for subplots
        rows, cols = random.choice([(1, 1), (1, 2), (1, 3), (2, 2)])


        _, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows))
        axes_list = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

        subplots_data = []

        for i, ax in enumerate(axes_list):
            plot_func = random.choice(
                [self._plot_random_scatter_on_ax, self._plot_random_boxplot_on_ax]
            )
            subplot_labels, x_label, y_label = plot_func(ax)
            subplot_data = {
                "subplot_index": i,
                "coordinates": subplot_labels,
                "x_label": x_label,
                "y_label": y_label,
            }
            subplots_data.append(subplot_data)

        # Save the figure
        plt.tight_layout()
        plt.savefig(image_path, bbox_inches="tight")
        plt.close()

        # Save the ground truth coordinates and axis labels
        output_data = {"subplots": subplots_data}
        with open(groundtruth_path, "w") as f:
            json.dump(output_data, f, indent=2)
