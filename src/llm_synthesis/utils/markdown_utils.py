import re
from re import Pattern


def remove_figs(text: str) -> str:
    """
    Function which removes markdown figures from extracted text papers

    Args:
        paper (str): The paper to remove figures from.

    Returns:
        str: The paper with figures removed.
    """

    fig_pattern: Pattern = re.compile(
        r"!\[(?:fig|image)\]\([^\)]*\)", re.IGNORECASE
    )

    # Remove all inline FIG_PATTERN matches
    cleaned = fig_pattern.sub("", text)

    return cleaned
