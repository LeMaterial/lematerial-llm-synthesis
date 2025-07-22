"""Load manual annotations from a json flile."""

import json
import os


def get_manual_annotations(directory_name: str = "annotations/") -> list[dict]:
    manual_annotations = []
    # Loop through all subdirectories in directory_name, get .json files
    for subdir in os.listdir(directory_name):
        if os.path.isdir(os.path.join(directory_name, subdir)):
            for file in os.listdir(os.path.join(directory_name, subdir)):
                if file.endswith(".json"):
                    manual_annotations.append(
                        load_manual_annotation(
                            os.path.join(directory_name, subdir, file)
                        )
                    )
    return manual_annotations


def load_manual_annotation(file_path: str) -> list[dict]:
    with open(file_path) as f:
        return json.load(f)


def main():
    manual_annotations = get_manual_annotations()
    print(manual_annotations[0][0].keys())


if __name__ == "__main__":
    main()
