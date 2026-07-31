import os
import sys
from huggingface_hub import HfApi

TOKEN = "ROTATED_TOKEN_REDACTED"
REPO_ID = "Chrishao/rfss"
DATA_DIR = "/Users/hc/Documents/research/Projects/dataset_RFSS/data"

api = HfApi(token=TOKEN)

files = [
    ("rfss_single.h5", "data/rfss_single.h5"),
    ("rfss_dataset.h5", "data/rfss_dataset.h5"),
]

for local_name, repo_path in files:
    local_path = os.path.join(DATA_DIR, local_name)
    size_gb = os.path.getsize(local_path) / 1e9
    print(f"Uploading {local_name} ({size_gb:.1f} GB) -> {repo_path}", flush=True)
    try:
        url = api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=repo_path,
            repo_id=REPO_ID,
            repo_type="dataset",
        )
        print(f"Done: {url}", flush=True)
    except Exception as e:
        print(f"ERROR uploading {local_name}: {e}", flush=True)
        sys.exit(1)

print("All uploads complete.", flush=True)
