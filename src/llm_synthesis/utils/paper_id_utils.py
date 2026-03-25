"""
Paper ID normalization for cond-mat: HF uses slash, filesystem uses period.
"""


def hf_id_to_folder_id(hf_id: str) -> str:
    """
    Convert HF dataset id to filesystem-safe folder id (cond-mat only).

    HF uses e.g. cond-mat/0503432; we use cond-mat.0503432 for folder names.
    """
    if isinstance(hf_id, str) and hf_id.startswith("cond-mat/"):
        return hf_id.replace("/", ".", 1)
    return hf_id


def folder_id_to_hf_id(paper_id: str) -> str:
    """
    Convert filesystem folder id to HF dataset id (cond-mat only).

    Folder names use cond-mat.0503432; HF uses cond-mat/0503432.
    """
    if isinstance(paper_id, str) and paper_id.startswith("cond-mat."):
        return paper_id.replace(".", "/", 1)
    return paper_id
