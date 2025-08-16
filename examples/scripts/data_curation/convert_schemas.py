import argparse

from datasets import (
    DatasetDict,
    Features,
    get_dataset_config_names,
    get_dataset_split_names,
    load_dataset,
)

from llm_synthesis.services.storage.paper_schema import schema as paper_schema
from llm_synthesis.services.storage.synthesis_schema import (
    schema as synthesis_schema,
)


def cast_schema(args):
    # get correct schema
    schema = paper_schema if "Papers" in args.dataset else synthesis_schema

    # get all configs
    configs = (
        [args.config] if args.config else get_dataset_config_names(args.dataset)
    )

    dataset_splits = {}

    # iterate through selected configs
    for config in configs:
        # get all splits in this config
        splits = (
            [args.split]
            if args.split
            else get_dataset_split_names(args.dataset, config)
        )

        # iterate through all selected splits
        for split in splits:
            loaded_dataset = load_dataset(
                args.dataset, name=config, split=split
            )

            casted_dataset = loaded_dataset.cast(
                Features.from_arrow_schema(schema)
            )
            dataset_splits[split] = casted_dataset

        if args.write_to_hub:
            dataset_dict = DatasetDict(dataset_splits)
            print(f"Pushing to hub: {args.dataset}")
            dataset_dict.push_to_hub(
                args.dataset, config_name=config, create_pr=True
            )
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--write-to-hub", action="store_true", help="")
    parser.add_argument(
        "--dataset",
        type=str,
        default="LeMaterial/LeMat-Synth-Papers",
        help=(
            "For the papers dataset, use 'LeMaterial/LeMat-Synth-Papers'. "
            "For the synthesis dataset, use 'LeMaterial/LeMat-Synth'. "
            "Note, that depending on which one you choose, the schema will "
            "be automatically inferred from `paper_schema.py` or "
            "`synthesis_schema.py`."
        ),
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="If None, this will run through all subsets.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="If None, this will run through all splits in specified subset.",
    )
    args = parser.parse_args()
    cast_schema(args)
