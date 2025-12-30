import pickle

from datasets import load_dataset


# Query function, takes in the split name, which column to check and a list of
# keywords to return the IDs of the papers that match
def query_db(dataset, split_name, text_column, list_keywords):
    ds = dataset[split_name]

    results = {}  # store keyword and corresponding IDs

    for keyword in list_keywords:
        # define filter function for this keyword
        def contains_keyword(example, kw=keyword):
            return (
                example[text_column] is not None
                and kw.lower() in example[text_column].lower()
            )

        # filter dataset
        filtered = ds.filter(contains_keyword)

        # extract IDs
        ids = filtered["id"]

        # store in dictionary
        results[keyword] = ids

    # Print results
    for kw, ids in results.items():
        print(f"\nKeyword: {kw}")
        print(f"Found {len(ids)} entries")
        print(ids)

    return results

    # From a given subset of papers, removes redundancies


def return_nonredundant_ids(results):
    all_ids = []

    for kw, ids in results.items():
        all_ids.extend(ids)

    # Convert to a set to remove duplicates
    unique_ids = set(all_ids)

    # Count them
    print("Number of papers containing at least one keyword:", len(unique_ids))

    return unique_ids


if __name__ == "__main__":
    # Load the "full" configuration and split
    dataset = load_dataset(
        "LeMaterial/LeMat-Synth-Papers",
        "full",  # configuration/subset
        split=None,  # other splits include 'arxiv', 'omg24', 'chemrxiv'
        token=True,
    )
    print("Loaded dataset with columns:", dataset.column_names)

    split_name_list = ["arxiv", "omg24", "chemrxiv"]
    text_column = "abstract"
    list_keywords = [
        "heterogeneous catalysis",
        "heterogeneous catalyst",
        "was heated at",
        "was heated under",
        "thermal treatment",
        "was measured at different temperatures",
        "activation energy",
        "variations",
        "conversion",
        "efficiency",
        "activation energy",
    ]
    """list_keywords = ["catalyst", "catalysis", "catalytic", "TOF", 
    "activation energy"]"""
    db = {}

    for split_name in split_name_list:
        result = query_db(
            dataset=dataset,
            split_name=split_name,
            text_column=text_column,
            list_keywords=list_keywords,
        )
        nonredundant_ids = return_nonredundant_ids(result)
        print(split_name, len(nonredundant_ids), nonredundant_ids)

        db[split_name] = nonredundant_ids

    # export db to pickle file
    with open("results/db_thermocatalysis.pkl", "wb") as f:
        pickle.dump(db, f)

    print("Saved db.pkl with keys:", list(db.keys()))
