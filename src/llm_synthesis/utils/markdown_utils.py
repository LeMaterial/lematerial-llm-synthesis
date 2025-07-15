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
    Remove references and all subsequent text from extracted papers.
    """

    # This pattern matches the reference heading
    # and then everything (.*) until the end of the string ($)
    reference_pattern: Pattern = re.compile(
        r"(# References|## References|### References).*$",
        re.IGNORECASE | re.DOTALL,
    )
    # re.DOTALL is crucial here to make '.' match newlines as well.

    # Remove everything in the string after the reference pattern
    cleaned = re.sub(reference_pattern, "", text)
    return cleaned


def clean_text(text: str) -> str:
    """
    Function which cleans the text by removing figures and references
    """
    cleaned = remove_figs(text)
    cleaned = remove_references(cleaned)
    return cleaned
