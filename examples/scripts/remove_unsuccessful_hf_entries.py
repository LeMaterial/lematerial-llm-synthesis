import logging

from datasets import Dataset, load_dataset

logging.basicConfig(level=logging.INFO)


def main():
    dataset = load_dataset("LeMaterial/LeMat-Synth")
    splits = dataset.keys()

    for split in splits:
        cts = 0
        df = dataset[split].to_pandas()
        len_of_df_before = len(df)
        for index, row in df.iterrows():
            if row["synthesized_material"] == "No materials synthesized":
                cts += 1
                df = df.drop(index)
        dataset[split] = Dataset.from_pandas(df)
        perc = cts / len_of_df_before * 100
        logging.info(f"Split: {split}")
        logging.info(f"Number of unsuccessful entries: {cts} ({perc:.2f}%)")

    dataset.push_to_hub("LeMaterial/LeMat-Synth", create_pr=True)


if __name__ == "__main__":
    main()
