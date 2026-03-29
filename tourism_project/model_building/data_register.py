
from huggingface_hub import upload_file
import os

HF_TOKEN = os.getenv("HF_TOKEN")

upload_file(
    path_or_fileobj="tourism_project/data/tourism.csv",
    path_in_repo="tourism.csv",
    repo_id="gayathri1909/tourism.csv",
    repo_type="dataset",
    token=HF_TOKEN
)

print("Dataset uploaded successfully")
