LINE_CHART_PROMPT = """
You will be provided with a line chart. The chart may not be chunked very well, 
so you may need to read only the plot in the center of the image.
In the chart, there will be several lines representing different data series.

1. Identify the different lines by their colors and labels.
2. For each line, extract the coordinates of the points that make up the line. 
Do not include any points that are not part of the line.
3. If the chart has metadata such as a title, x-axis label, y-axis labels, 
or units, extract that information as well. 
Keep the scientific terms in Markdown format.
4. Output the data in the specified format:

Name_of_Line_1: [[x1, y1], [x2, y2], ...]
title:
x_axis_label:
x_axis_unit:
y_left_axis_label:
y_left_axis_unit:

Do not output any other text, just the data in the format above.
"""
