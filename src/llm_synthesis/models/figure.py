from pydantic import BaseModel


class FigureInfo(BaseModel):
    """Information about a figure found in markdown text."""

    base64_data: str
    alt_text: str
    position: int  # Character position in the text
    context_before: str
    context_after: str
    figure_reference: str  # e.g., "Figure 2", "Fig. 3a", etc.
    # figure_class: str | None = None # TODO: add fig class


class FigureInfoWithPaper(FigureInfo):
    """Information about figure found in markdown text with the paper text."""

    paper_text: str
    si_text: str
