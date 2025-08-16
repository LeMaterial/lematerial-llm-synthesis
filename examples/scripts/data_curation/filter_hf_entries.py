import logging
import re

from datasets import load_dataset

logging.basicConfig(level=logging.INFO)


def is_valid_material_name(name):
    """
    Check if a material name is valid based on post-processing criteria.
    Returns False if the name should be filtered out.
    """
    if not isinstance(name, str):
        return False

    # Convert to lowercase for pattern matching
    name_lower = name.lower().strip()

    # Filter out empty names
    if not name_lower:
        return False

    # Filter out "No materials synthesized"
    if name_lower == "no materials synthesized":
        return False

    # Filter out names that are only numbers (any length)
    if re.match(r"^\d+$", name_lower):
        return False

    # Filter out names that are only 1-2 symbols long (single character/number)
    if len(name_lower) <= 2:
        return False

    # Filter out names containing "intermediate" followed by numbers
    # Matches: "intermediate 1", "intermediate2", "intermediate 123", etc.
    if re.search(r"intermediate\s*\d+", name_lower):
        return False

    # Filter out names containing "compound" followed by numbers
    # Matches: "compound 1", "compound2", "compound 123", etc.
    if re.search(r"compound\s*\d+", name_lower):
        return False

    # Filter out names with pattern {numbers}{letters} like "1a", "23bc"
    # This catches simple numbered compounds with letter suffixes
    if re.match(r"^\d+[a-z]+$", name_lower):
        return False

    # Filter out names with pattern {letters}{numbers} like "a1", "abc23"
    # This catches lettered compounds with number suffixes
    if re.match(r"^[a-z]+\d+$", name_lower):
        return False

    # Additional patterns that might indicate intermediate/generic names:

    # Filter out names that are just "product" followed by numbers
    if re.search(r"product\s*\d+", name_lower):
        return False

    # Filter out names that are just "material" followed by numbers
    if re.search(r"material\s*\d+", name_lower):
        return False

    # Filter out names that are just "sample" followed by numbers
    if re.search(r"sample\s*\d+", name_lower):
        return False

    return True


def filter_dataset_entry(example):
    """
    Filter function for dataset entries.
    Returns True if the entry should be kept, False if it should be removed.
    """
    synthesized_material = example.get("synthesized_material", "")
    return is_valid_material_name(synthesized_material)


def main():
    dataset = load_dataset("LeMaterial/LeMat-Synth", name="full")
    splits = dataset.keys()

    total_removed_by_category = {
        "no_materials_synthesized": 0,
        "only_numbers": 0,
        "single_character": 0,
        "intermediate_pattern": 0,
        "compound_pattern": 0,
        "number_letter_pattern": 0,
        "generic_names": 0,
        "other": 0,
    }

    for split in splits:
        original_length = len(dataset[split])

        # Get detailed statistics before filtering
        removed_stats = analyze_removed_entries(dataset[split])

        # Update total statistics
        for key in total_removed_by_category:
            total_removed_by_category[key] += removed_stats[key]

        # Filter out entries with invalid material names
        dataset[split] = dataset[split].filter(filter_dataset_entry)

        filtered_length = len(dataset[split])
        removed_count = original_length - filtered_length
        perc = (
            removed_count / original_length * 100 if original_length > 0 else 0
        )

        logging.info(f"\nSplit: {split}")
        logging.info(f"Original entries: {original_length}")
        logging.info(f"Filtered entries: {filtered_length}")
        logging.info(f"Removed entries: {removed_count} ({perc:.2f}%)")

        # Log detailed breakdown
        logging.info("Breakdown of removed entries:")
        for category, count in removed_stats.items():
            if count > 0:
                category_perc = (
                    count / original_length * 100 if original_length > 0 else 0
                )
                logging.info(
                    f"  - {category.replace('_', ' ').title()}: {count} ({category_perc:.2f}%)"  # noqa: E501
                )

    # Log overall statistics
    logging.info("\n=== OVERALL STATISTICS ===")
    total_original = sum(
        len(dataset[split]) + sum(removed_stats.values())
        for split in splits
        for removed_stats in [analyze_removed_entries(dataset[split])]
    )
    total_removed = sum(total_removed_by_category.values())

    logging.info("Total removed entries by category:")
    for category, count in total_removed_by_category.items():
        if count > 0:
            logging.info(f"  - {category.replace('_', ' ').title()}: {count}")

    logging.info(f"Total removed entries: {total_removed}")
    logging.info(f"Total original entries: {total_original}")
    logging.info(f"Total filtered entries: {total_original - total_removed}")

    # Push the filtered dataset back to hub
    dataset.push_to_hub("LeMaterial/LeMat-Synth", create_pr=True)


def analyze_removed_entries(dataset_split):
    """Analyze which entries would be removed and categorize them."""
    stats = {
        "no_materials_synthesized": 0,
        "only_numbers": 0,
        "single_character": 0,
        "intermediate_pattern": 0,
        "compound_pattern": 0,
        "number_letter_pattern": 0,
        "generic_names": 0,
        "other": 0,
    }

    for example in dataset_split:
        name = example.get("synthesized_material", "")
        if not isinstance(name, str):
            stats["other"] += 1
            continue

        name_lower = name.lower().strip()

        if not name_lower:
            stats["other"] += 1
        elif name_lower == "no materials synthesized":
            stats["no_materials_synthesized"] += 1
        elif re.match(r"^\d+$", name_lower):
            stats["only_numbers"] += 1
        elif len(name_lower) == 1:
            stats["single_character"] += 1
        elif re.search(r"intermediate\s*\d+", name_lower):
            stats["intermediate_pattern"] += 1
        elif re.search(r"compound\s*\d+", name_lower):
            stats["compound_pattern"] += 1
        elif re.match(r"^\d+[a-z]+$", name_lower) or re.match(
            r"^[a-z]+\d+$", name_lower
        ):
            stats["number_letter_pattern"] += 1
        elif (
            re.search(r"product\s*\d+", name_lower)
            or re.search(r"material\s*\d+", name_lower)
            or re.search(r"sample\s*\d+", name_lower)
        ):
            stats["generic_names"] += 1
        elif not is_valid_material_name(name):
            stats["other"] += 1

    return stats


if __name__ == "__main__":
    main()
