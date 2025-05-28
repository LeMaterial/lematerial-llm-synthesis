import dspy


class TextToParagraphSignature(dspy.Signature):
    """Signature for the full text parser"""

    publication_text: str = dspy.InputField(
        description="The main text of the publication which contains the synthesis paragraphs."
    )
    si_text: str | None = dspy.InputField(
        description="The supporting information for the publication which may contain additional synthesis paragraphs."
    )
    synthesis_paragraphs: str = dspy.OutputField(
        description="The paragraphs containing the synthesis procedures of all compounds synthesized in the publication."
    )
