import matplotlib.pyplot as plt

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
