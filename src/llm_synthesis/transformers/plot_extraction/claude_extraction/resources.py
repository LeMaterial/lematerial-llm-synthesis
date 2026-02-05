# Legacy prompt without figure context
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

# New prompt with figure context - use this when figure context is available
LINE_CHART_PROMPT_WITH_CONTEXT = """
You will be provided with a line chart from a scientific paper.

FIGURE CONTEXT FROM PAPER:
{figure_context}

The context above may contain the figure caption with important information including:
- Symbol-to-material mappings (e.g., "(▲) Co₇Mo₃/MCM-41, (○) Co₇Mo₃/SiO₂")
- What each curve/line represents
- Experimental conditions

SERIES NAMING:
Your goal is to name each series so it can be linked back to the specific material
or sample it represents.

IMPORTANT: Chemical formulas with numbers/subscripts are often misread from plot
images. The figure context from the paper text is more reliable for exact names.
However, you must VERIFY each match - do not blindly assign context names to curves.

1. FIRST check the context for material names listed in the figure caption
2. For each curve in the image, identify its visual marker (symbol shape, color,
   line style) and try to match it to a specific material from the context
3. Only use a context name if you can CONFIRM the match through:
   - Symbol mapping in the caption (e.g., "(▲) MaterialA")
   - Matching the legend text in the image to a context name (even if the image
     text is slightly garbled, e.g., image shows "Pt50Cu50" and context has
     "Pt25Cu75" - these are DIFFERENT materials, do not substitute)
   - Consistent ordering between legend and caption when symbols clearly correspond
4. If you CANNOT verify which context name belongs to which curve, use a visual
   description (color, marker shape) instead - the downstream linker can handle
   approximate matching, but WRONG assignments cannot be corrected
5. Never assume that the order of materials in the context matches the order of
   lines in the image

No data is better than wrong data.

Instructions:
1. Match each line/curve in the image to its identifier from the context or legend.
2. For each line, extract the coordinates of the data points.
   Do not include points that are not part of the line.
3. Extract axis metadata (title, labels, units) if visible.
   Keep scientific terms in Markdown format.
4. Output in this exact format:

Series_Name: [[x1, y1], [x2, y2], ...]
title:
x_axis_label:
x_axis_unit:
y_left_axis_label:
y_left_axis_unit:

Do not output any other text, just the data in the format above.
"""
