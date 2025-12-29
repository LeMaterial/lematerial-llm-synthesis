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


def remove_references(text: str) -> str:
    """
    Remove 50 lines after references section from extracted papers.
    """

    # This pattern matches the reference heading and captures the next 50 lines
    reference_pattern: Pattern = re.compile(
        r"(# References|## References|### References)(?:.*\n){0,50}",
        re.IGNORECASE,
    )

    # Remove just the references section and 50 lines after
    cleaned = re.sub(reference_pattern, "", text)
    return cleaned


def clean_text(text: str) -> str:
    """
    Function which cleans the text by removing figures and references
    """
    cleaned = remove_figs(text)
    cleaned = remove_references(cleaned)
    return cleaned
