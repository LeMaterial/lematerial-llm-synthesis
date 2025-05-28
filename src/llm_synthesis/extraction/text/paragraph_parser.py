import dspy
from llm_synthesis.extraction.text.signatures import TextToParagraphSignature
from llm_synthesis.utils.markdown_utils import remove_figs


class ParagraphParser(dspy.Module):
    """
    Class which parses the text of a publication and returns a list of paragraphs containing the synthesis procedure.

    Args:
        remove_figs (bool): Whether to remove figures from the text.
    """

    def __init__(
        self,
        signature: dspy.Signature = TextToParagraphSignature,
        remove_figs: bool = True,
    ):
        self.remove_figs = remove_figs
        self.predict = dspy.Predict(signature)

    def __call__(self, publication_text: str, si_text: str = "") -> str:
        publication_text = self._clean_text(publication_text)
        if si_text:
            si_text = self._clean_text(si_text)
        # TODO: potentially make common functions/module for this
        return self.predict(publication_text=publication_text, si_text=si_text)

    def _clean_text(self, text: str) -> str:
        if self.remove_figs:
            text = remove_figs(text)
        return text
