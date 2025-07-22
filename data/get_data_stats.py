from huggingface_hub import HfApi

token = 'YOUR_HF_TOKEN'

# Initialize API
api = HfApi(token=token)

# Dataset repo ID
repo_id = "LeMaterial/LeMat-Synth"

# Get dataset info with file metadata
dataset_info = api.dataset_info(repo_id=repo_id, files_metadata=True)

# Calculate total size in bytes
total_size_bytes = sum(file.size for file in dataset_info.siblings if file.size)

# Convert to MB and GB
total_size_mb = total_size_bytes / (1024 * 1024)
total_size_gb = total_size_mb / 1024

usedsotrage = dataset_info.usedStorage/(1024*1024*1024)

print(f"Total size of parquet files: {total_size_mb:.2f} MB ({total_size_gb:.2f} GB)")
print(f"Total storage used: {usedsotrage:.2f} GB")
