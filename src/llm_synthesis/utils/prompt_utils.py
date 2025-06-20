def read_prompt_str_from_txt(prompt_path: str) -> str:
    """
    Read a prompt from a text file.

    Args:
        prompt_path: The path to the prompt file.

    Returns:
        The prompt string.
    """
    with open(prompt_path, "r") as f:
        return f.read()