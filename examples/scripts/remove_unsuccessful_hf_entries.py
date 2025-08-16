import logging

from datasets import load_dataset

logging.basicConfig(level=logging.INFO)


def main():
    dataset = load_dataset("LeMaterial/LeMat-Synth", name="full")
    splits = dataset.keys()

    for split in splits:
        original_length = len(dataset[split])

        # Filter out unsuccessful entries using the filter method
        dataset[split] = dataset[split].filter(
            lambda example: example["synthesized_material"]
            != "No materials synthesized"
        )

        filtered_length = len(dataset[split])
        removed_count = original_length - filtered_length
        perc = removed_count / original_length * 100

        logging.info(f"Split: {split}")
        logging.info(
            f"Number of unsuccessful entries: {removed_count} ({perc:.2f}%)"
        )

    dataset.push_to_hub("LeMaterial/LeMat-Synth", create_pr=True)


if __name__ == "__main__":
    main()
