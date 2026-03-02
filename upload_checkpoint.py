import os
from huggingface_hub import HfApi, create_repo

# Define repositories and path
repo_id = "Phonsiri/gemma-2-2b-GRPO-Reasoning"
# You can change the 'checkpoint-10' string below to whatever checkpoint you want to push!
folder_path = "./grpo_output/checkpoint-10"

print(f"Uploading {folder_path} to Hugging Face Hub: {repo_id}...")

api = HfApi()

# Create the repository if it doesn't exist
try:
    create_repo(repo_id, repo_type="model", exist_ok=True)
    print(f"Verified/Created repository: {repo_id}")
except Exception as e:
    print(f"Note on repo creation: {e}")

api.upload_folder(
    folder_path=folder_path,
    repo_id=repo_id,
    path_in_repo=os.path.basename(folder_path.rstrip("/")), # e.g. "checkpoint-10"
    repo_type="model",
)

print(f"Upload Complete! Check it out at https://huggingface.co/{repo_id}/tree/main")
