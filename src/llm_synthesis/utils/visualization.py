import matplotlib.pyplot as plt
import numpy as np

from llm_synthesis.models.plot import ExtractedLinePlotData


def visualize_chart(data: ExtractedLinePlotData, figure_class: str):
    if figure_class == "Line plot":
        visualize_line_chart(data)
    elif figure_class == "Bar plot":
        visualize_bar_chart(data)
    elif figure_class == "Scatter plot":
        visualize_scatter_plot(data)
    else:
        raise ValueError(f"Unsupported figure class: {figure_class}")


def visualize_line_chart(data: ExtractedLinePlotData):
    names = list(data.name_to_coordinates.keys())

    markers = [
        "o",
        "^",
        "s",
        "D",
        "v",
        "<",
        ">",
        "p",
        "*",
        "h",
        "H",
        "+",
        "x",
        "|",
        "_",
    ]

    for i, name in enumerate(names):
        coords = data.name_to_coordinates[name]
        x, y = zip(*coords)
        marker = markers[i % len(markers)]
        plt.plot(x, y, label=name, marker=marker)

    xlabel = f"{data.x_axis_label}_({data.x_axis_unit})"
    ylabel = f"{data.y_left_axis_label}_({data.y_left_axis_unit})"

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(data.title)
    plt.legend()
    plt.grid()
    plt.show()


def visualize_line_chart_specialized(
    data: ExtractedLinePlotData, rmse: float = 0
):
    names = list(data.name_to_coordinates.keys())

    colors = [
        "#B20002",  # 10_Cs_FePc
        "#A84EFF",  # 2_Cs_FePc
        "#30FF24",  # 2_Cs_FePc+CoPc_9_1_
        "#1C61F4",  # 8_Ba_CoPc
        "#000000",  # Fe_KM1
    ]

    markers = [
        "^",
        "s",
        "o",
        "D",
        "o",
    ]

    plt.figure(figsize=(5, 4))

    for i, name in enumerate(names):
        print(name)
        coords = data.name_to_coordinates[name]
        x, y = zip(*coords)
        x = np.array(x)
        y = np.array(y)
        marker = markers[i % len(markers)]
        color = colors[i % len(colors)]
        plt.plot(x, y, label=name, marker=marker, color=color)
        # Plot confidence intervals based on normalized RMSE
        # y axis only
        if rmse > 0:
            # Convert normalized RMSE back to y-axis units
            y_scale = 10000 - 0
            rmse_y_units = rmse * y_scale
            y_standard_error = rmse_y_units / np.sqrt(len(x))
            y_ci_95 = 1.96 * y_standard_error

            # x_scale = 500 - 200
            # rmse_x_units = rmse * x_scale
            # x_standard_error = rmse_x_units / np.sqrt(len(x))
            # x_ci_95 = 1.96 * x_standard_error  # NOT USED FOR NOW

            # 95% confidence interval in y axis units
            plt.fill_between(
                x, y - y_ci_95, y + y_ci_95, alpha=0.1, color=color
            )

    xlabel = f"{data.x_axis_label}_({data.x_axis_unit})"
    ylabel = f"{data.y_left_axis_label}_({data.y_left_axis_unit})"

    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlim(200, 500)
    plt.ylim(0, 10000)
    plt.xticks(np.arange(200, 550, 50))
    plt.yticks(np.arange(0, 12000, 2000))

    plt.title(data.title, fontsize=12)
    plt.legend(fontsize=12, frameon=False)
    plt.show()


def visualize_bar_chart(data: ExtractedLinePlotData):
    names = list(data.name_to_coordinates.keys())

    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]

    # Extract all unique x-values to use as categories
    all_x_values = set()
    for coords in data.name_to_coordinates.values():
        for x, y in coords:
            all_x_values.add(x)

    x_categories = sorted(list(all_x_values))

    # Set up bar positions
    bar_width = 0.8 / len(names) if len(names) > 1 else 0.6
    x_positions = np.arange(len(x_categories))

    for i, name in enumerate(names):
        coords = data.name_to_coordinates[name]

        # Create a mapping of x-values to y-values for this series
        coord_dict = {x: y for x, y in coords}

        # Get y-values for each x-category (0 if missing)
        y_values = [coord_dict.get(x, 0) for x in x_categories]

        # Calculate bar positions for this series
        if len(names) > 1:
            bar_positions = x_positions + (i - (len(names) - 1) / 2) * bar_width
        else:
            bar_positions = x_positions

        color = colors[i % len(colors)]
        plt.bar(
            bar_positions,
            y_values,
            bar_width,
            label=name,
            color=color,
            alpha=0.8,
        )

    # Set labels and formatting
    xlabel = f"{data.x_axis_label}_({data.x_axis_unit})"
    ylabel = f"{data.y_left_axis_label}_({data.y_left_axis_unit})"

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(data.title)
    plt.xticks(x_positions, [str(x) for x in x_categories])
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


def visualize_scatter_plot(data: ExtractedLinePlotData):
    names = list(data.name_to_coordinates.keys())

    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]

    markers = [
        "o",
        "^",
        "s",
        "D",
        "v",
        "<",
        ">",
        "p",
        "*",
        "h",
        "H",
        "+",
        "x",
        "|",
        "_",
    ]

    for i, name in enumerate(names):
        coords = data.name_to_coordinates[name]
        if coords:  # Check if coordinates exist
            x, y = zip(*coords)
            color = colors[i % len(colors)]
            marker = markers[i % len(markers)]
            plt.scatter(
                x,
                y,
                label=name,
                color=color,
                marker=marker,
                s=50,
                alpha=0.7,
                edgecolors="black",
                linewidth=0.5,
            )

    # Set labels and formatting
    xlabel = f"{data.x_axis_label}_({data.x_axis_unit})"
    ylabel = f"{data.y_left_axis_label}_({data.y_left_axis_unit})"

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(data.title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
