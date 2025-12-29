import json
import re
from pathlib import Path

# import tiktoken
import transformers
from datasets import load_dataset
from llm import LLM  # your local wrapper
from tqdm import tqdm

################################################################################
# ---------------------------  Helper utilities  ----------------------------- #
################################################################################


def split_text_into_chunks(
    text: str, max_tokens: int, tokenizer: transformers.AutoTokenizer
) -> list[str]:
    """Split text into chunks based on token count, trying to break at sentence boundaries."""
    sentences = text.split(". ")
    chunks = []
    current_chunk = []
    current_token_count = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        sentence_token_count = len(tokenizer.encode(sentence))

        if (
            current_token_count + sentence_token_count > max_tokens
            and current_chunk
        ):
            # Current chunk is full, save it and start a new one
            chunks.append(". ".join(current_chunk) + ".")
            current_chunk = [sentence]
            current_token_count = sentence_token_count
        else:
            current_chunk.append(sentence)
            current_token_count += sentence_token_count

    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append(". ".join(current_chunk) + ".")

    return chunks


################################################################################
# ----------------------------  Core analysis  ------------------------------- #
################################################################################


def _call_llm(chunk: str, client: LLM) -> dict:
    """Send *chunk* to the LLM and parse its JSON answer."""
    prompt = f"""Analyze the following text and answer the questions in JSON format:


{chunk}

Questions:
1. Does it contain a material synthesis recipe? (Answer with true or false)
2. If yes, what is the material name? (Answer with the material name or "N/A" if no recipe)
3. If yes, which category of materials does it belong to? (Answer with the specific material type or "N/A" if no recipe)
    List of material categories:
    Metals, Ceramics, Semiconductors, Superconductors, Composites, Biomaterials, Nanomaterials, Polymers, Magnetic, Textiles, Chemicals, Other

Format your response as a JSON object with the following structure:
{{
    "contains_recipe": true/false,
    "material_name": "material name or N/A",
    "material_category": "material category or N/A"
}}
"""

    response = client.generate_text(
        prompt, response_format={"type": "json_object"}
    )

    try:
        response_cleaned = (
            response.strip().strip("```json").strip("```").strip()
        )
        return json.loads(response_cleaned)
    except Exception:
        return {
            "contains_recipe": False,
            "material_name": "N/A",
            "material_category": "N/A",
        }


def analyze_article(
    text: str,
    client: LLM,
    max_tokens: int,
    tokenizer: transformers.AutoTokenizer,
) -> dict:
    """Analyze a full article (can be long) and merge chunk-level answers."""
    # strip MD images which confuse the model
    text = re.sub(r"!\[(Image|fig)\]\([^)]*\)", " ![\g<1>] ", text)

    chunks = split_text_into_chunks(text, max_tokens, tokenizer)
    # print("lenght of chunks", [len(tokenizer.encode(ch)) for ch in chunks])
    iterator = (
        tqdm(chunks, desc="Analyzing text chunks", leave=False)
        if len(chunks) > 1
        else chunks
    )

    answers = []
    for chunk in iterator:
        answer = _call_llm(chunk, client)
        answers.append(answer)

    result = {
        "contains_recipe": False,
        "material_name": "N/A",
        "material_category": "N/A",
    }
    for ans in answers:
        if ans.get("contains_recipe") == True:
            print("--> material_name", ans.get("material_name"))
            print("--> material_category", ans.get("material_category"))
            result["contains_recipe"] = True
            if ans.get("material_name") != "N/A":
                if isinstance(ans["material_name"], list):
                    result["material_name"] = ",".join(ans["material_name"])
                else:
                    result["material_name"] = ans["material_name"]

                if isinstance(ans["material_category"], list):
                    result["material_category"] = ans["material_category"][0]
                else:
                    result["material_category"] = ans["material_category"]
            break
    return result


################################################################################
# -----------------------  HF dataset processing entry  ---------------------- #
################################################################################


def process_hf_dataset(
    dataset_id: str,
    output_dir: str,
    client: LLM,
    split: str = "train",
    text_column: str = "text_paper",
    batch_size: int = 1,
    max_tokens: int = 120000,
    model: str = "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
) -> Path:
    """Load a 🤗 dataset, append three new columns, and save with `save_to_disk()`."""
    print(f"Loading dataset '{dataset_id}' ({split} split)…")
    ds = (
        load_dataset(
            "json", data_files=str(Path(dataset_id).expanduser()), split=split
        )
        if Path(dataset_id).expanduser().exists()
        else load_dataset(dataset_id, split=split)
    )

    if text_column not in ds.column_names:
        raise ValueError(
            f"Column '{text_column}' not found. Available: {ds.column_names}"
        )

    print(f"→ {len(ds):,} rows. Starting analysis…")

    tokenizer = transformers.AutoTokenizer.from_pretrained(model)

    def _annotate(batch):
        analytics = [
            analyze_article(txt, client, max_tokens, tokenizer)
            for txt in batch[text_column]
        ]
        return {
            "contains_recipe": [a["contains_recipe"] for a in analytics],
            "material_name": [a["material_name"] for a in analytics],
            "material_category": [a["material_category"] for a in analytics],
        }

    ds = ds.map(
        _annotate,
        batched=True,
        batch_size=batch_size,
        desc="🔍 Detecting synthesis recipes",
    )

    out_path = Path(output_dir).expanduser().absolute()
    print(f"Saving enriched dataset to {out_path} …")
    ds.save_to_disk(str(out_path))
    print("✅ Done. Reload anytime with `load_from_disk`. 🌟")
    return out_path


################################################################################
# ---------------------------------  CLI  ------------------------------------ #
################################################################################

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Add recipe-detection columns to a 🤗 dataset with a live progress bar."
        )
    )
    parser.add_argument(
        "dataset", help="HF dataset ID or local `load_from_disk` path"
    )
    parser.add_argument(
        "output_dir", help="Where to save the processed dataset"
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--text_column", default="text_paper")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--model", default="mistralai/Mistral-Small-3.1-24B-Instruct-2503"
    )
    parser.add_argument("--provider", default="vllm")
    parser.add_argument("--max_tokens", type=int, default=120000)

    args = parser.parse_args()

    try:
        llm_client = LLM(
            model_name=args.model,
            provider=args.provider,
            port=args.port,
        )
    except Exception as e:
        raise SystemExit(f"Could not init LLM: {e}")

    process_hf_dataset(
        dataset_id=args.dataset,
        output_dir=args.output_dir,
        client=llm_client,
        split=args.split,
        text_column=args.text_column,
        batch_size=args.batch_size,
        max_tokens=args.max_tokens,
        model=args.model,
    )
