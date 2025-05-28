import re
from typing import Pattern


def remove_figs(text: str) -> str:
    """
    Function which removes markdown figures from extracted text papers

    Args:
        paper (str): The paper to remove figures from.

    Returns:
        str: The paper with figures removed.
    """

    FIG_PATTERN: Pattern = re.compile(r"!\[(?:fig|image)\]\([^\)]*\)", re.IGNORECASE)

    # Remove all inline FIG_PATTERN matches
    cleaned = FIG_PATTERN.sub("", text)

    return cleaned
