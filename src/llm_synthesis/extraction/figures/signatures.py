import dspy


class FigureExtractionSignature(dspy.Signature):
    """Signature for the figure extraction parser."""

    publication_text: str = dspy.InputField(description="The text of the publication.")
    si_text: str = dspy.InputField(
        description="The text of the supporting information."
    )
    figure_bs64: str = dspy.InputField(
        description="The base64 encoded image of the figure to extract."
    )
    figure_description: str = dspy.OutputField(
        description="A detailed  description of the figure."
    )


class FigureDescriptionSignature(dspy.Signature):
    """
    Advanced signature for generating detailed scientific descriptions of figures in research papers.

    This signature is designed to handle various types of scientific figures including:
    - Plots, graphs, and charts (XY plots, bar charts, scatter plots, etc.)
    - Spectroscopy data (NMR, IR, XRD, UV-Vis, etc.)
    - Microscopy and imaging data (SEM, TEM, AFM, optical microscopy, etc.)
    - Schematic diagrams and experimental setups
    - Molecular structures and reaction schemes
    - Performance metrics and characterization data

    The system should ignore non-scientific figures like journal logos, author photos, etc.
    """

    publication_text: str = dspy.InputField(
        description="Complete text of the main publication containing context about the research, methodology, and results."
    )
    si_text: str = dspy.InputField(
        description="Supporting information text (optional) containing additional experimental details and supplementary data."
    )
    figure_base64: str = dspy.InputField(
        description="Base64 encoded image of the figure to analyze and describe."
    )
    caption_context: str = dspy.InputField(
        description="Text context surrounding the figure position including the figure caption and nearby paragraphs that reference this figure."
    )
    figure_position_info: str = dspy.InputField(
        description="Information about the figure's position in the document (e.g., 'Figure 2', 'Fig. 3a', 'Scheme 1') to help with contextual understanding."
    )

    figure_description: str = dspy.OutputField(
        description="""Generate a comprehensive scientific description of the figure following these guidelines:

ANALYSIS APPROACH:
1. First determine if this is a scientific figure worth describing (ignore logos, author photos, journal branding, etc.)
2. Identify the figure type (plot/graph, spectroscopy, microscopy, schematic, etc.)
3. Extract quantitative data and trends where visible
4. Connect observations to the research context from the paper

DESCRIPTION STRUCTURE:
- Start with figure type and main purpose
- Describe axes, scales, units, and data series for plots
- Report key quantitative values, trends, and comparisons
- Explain peak positions, intensities, and assignments for spectroscopy
- Describe morphology, scale bars, and structural features for imaging
- Connect findings to the broader research narrative

SCIENTIFIC RIGOR:
- Use precise scientific terminology appropriate to the field
- Include specific values, ranges, and units when visible
- Note experimental conditions and parameters shown
- Identify trends, correlations, and significant observations
- Maintain objectivity while being thorough

FORMAT: Provide a detailed paragraph description (100-300 words) that would be valuable for researchers understanding the figure without seeing it. If the figure is non-scientific (logo, etc.), respond with: "NON_SCIENTIFIC_FIGURE"

EXAMPLE OUTPUT STYLE:
"This X-ray diffraction pattern shows the crystalline structure of the synthesized catalyst, with the main diffraction peaks appearing at 2θ values of 26.5°, 33.8°, and 50.4°, corresponding to the (002), (101), and (110) planes of the hexagonal graphite structure. The sharp, intense peaks indicate high crystallinity, while the peak at 26.5° shows slight broadening suggesting some disorder in the graphitic layers. The absence of peaks below 20° confirms the removal of intercalated species during thermal treatment. Additional weak peaks at 43.2° and 77.5° are attributed to metallic copper nanoparticles (Cu(111) and Cu(220) reflections), consistent with the XPS analysis showing metallic copper content of approximately 15 wt%."
"""
    )
