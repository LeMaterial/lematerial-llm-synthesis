"""
Create a subset of dataset based on folder names in the annotations directory and upload to HuggingFace.
Configuration is read from config.yaml file.
"""

from datasets import load_dataset, Dataset, DatasetDict
import os
import yaml

# Load configuration
config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Extract configuration parameters
dataset_uri = config['dataset']['uri']
dataset_split = config['dataset']['split']
subset_name = config['output']['uri']
output_split = config['output']['split']
private_dataset = config['output']['private']
annotations_dir = config['annotations']['directory']

# Convert relative annotations directory to absolute path
if not os.path.isabs(annotations_dir):
    annotations_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", annotations_dir)

print(f"Configuration loaded:")
print(f"  Dataset: {dataset_uri}")
print(f"  Split: {dataset_split}")
print(f"  Subset name: {subset_name}")
print(f"  Annotations dir: {annotations_dir}")
print()

# Get paper IDs from annotations directory
paper_ids = []
if os.path.exists(annotations_dir):
    for item in os.listdir(annotations_dir):
        if os.path.isdir(os.path.join(annotations_dir, item)):
            paper_ids.append(item)
else:
    print(f"Error: Annotations directory not found: {annotations_dir}")

print(f"Found {len(paper_ids)} paper IDs in given directory: {annotations_dir}")

# Load the original dataset
print("Loading original dataset...")
dataset = load_dataset(dataset_uri, split=dataset_split)
print(f"Original dataset has {len(dataset)} entries")

# Filter dataset to only include papers with matching IDs
filtered_entries = []
for entry in dataset:
    dataset_id = entry.get('id', '')
    
    # Check direct match
    if dataset_id in paper_ids:
        filtered_entries.append(entry)
    # Check for cond-mat format conversion
    elif dataset_id.startswith('cond-mat/'):
        # Convert cond-mat/XXXX to cond-mat.XXXX for matching
        alt_format = dataset_id.replace('cond-mat/', 'cond-mat.')
        if alt_format in paper_ids:
            filtered_entries.append(entry)

print(f"\nFound {len(filtered_entries)} matching papers out of {len(paper_ids)} requested")

# Create new dataset with filtered entries
if len(filtered_entries) > 0:
    # Get all field names from the original dataset
    field_names = list(dataset.features.keys())
    
    # Create dictionary with all fields automatically
    subset_dict_data = {}
    for field in field_names:
        subset_dict_data[field] = [entry[field] for entry in filtered_entries]
    
    subset_dataset = Dataset.from_dict(subset_dict_data)
    
    # Create DatasetDict
    subset_dict = DatasetDict({
        output_split: subset_dataset
    })
    
    print("\nPushing dataset to Hugging Face...")
    
    # Push to Hugging Face with configured name
    subset_dict.push_to_hub(f"{subset_name}", private=private_dataset)
    
    print(f"Dataset pushed {subset_name}: {len(subset_dataset)} entries")
    
    # Save locally first for verification
    
else:
    print("No matching papers found. Check the ID format in the dataset vs annotations.")
