import matplotlib.pyplot as plt

from llm_synthesis.models.plot import ExtractedLinePlotData


def visulize_line_chart(data: ExtractedLinePlotData):
    names = list(data.name_to_coordinates.keys())
    for name in names:
        coords = data.name_to_coordinates[name]
        x, y = zip(*coords)
        plt.plot(x, y, label=name)

    xlabel = f"{data.x_axis_label}_({data.x_axis_unit})"
    ylabel = f"{data.y_left_axis_label}_({data.y_left_axis_unit})"

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(data.title)
    plt.legend()
    plt.grid()
    plt.show()
