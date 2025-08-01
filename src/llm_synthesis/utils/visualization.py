import matplotlib.pyplot as plt
import numpy as np

from llm_synthesis.models.plot import ExtractedLinePlotData


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

            # 95% confidence interval in y axis units
            ci_95 = 1.96 * rmse_y_units
            plt.fill_between(x, y - ci_95, y + ci_95, alpha=0.1, color=color)

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
