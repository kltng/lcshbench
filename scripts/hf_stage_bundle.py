# Stage code + v2.2 data into a private HF dataset repo the Job will pull from.
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
REPO = "kltng/lcsh-bench-v22-job"
api.create_repo(REPO, repo_type="dataset", private=True, exist_ok=True)

# 1) lcsh-benchmark package (src + pyproject + lock) — installable in the job
api.upload_folder(repo_id=REPO, repo_type="dataset",
    folder_path="/home/kltang/projects/lcsh-benchmark",
    path_in_repo="lcsh-benchmark",
    allow_patterns=["src/**", "pyproject.toml", "uv.lock", "README.md"],
    commit_message="stage lcsh-benchmark package")

# 2) lcsh_db_builder package (lcsh-onnx db-builder) — for the lcsh-onnx baseline
api.upload_folder(repo_id=REPO, repo_type="dataset",
    folder_path="/home/kltang/projects/lcsh/lcsh-onnx/db-builder",
    path_in_repo="db-builder",
    allow_patterns=["src/**", "pyproject.toml"],
    commit_message="stage lcsh_db_builder package")

# 3) v2.2 dataset (dev/test/subset)
api.upload_folder(repo_id=REPO, repo_type="dataset",
    folder_path="/home/kltang/projects/lcsh-benchmark/data/v2.2",
    path_in_repo="data/v2.2",
    commit_message="stage v2.2 dataset")

print("staged ->", f"https://huggingface.co/datasets/{REPO}")
