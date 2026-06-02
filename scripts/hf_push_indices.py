import os
from huggingface_hub import HfApi
api=HfApi(token=os.environ["HF_TOKEN"])
REPO="kltng/lcsh-bench-v22-indices"
api.create_repo(REPO,repo_type="dataset",private=True,exist_ok=True)
for name in ("text-embedding-3-small","embeddinggemma-300m-lcsh-merged-d256"):
    print("uploading",name,flush=True)
    api.upload_folder(repo_id=REPO,repo_type="dataset",
        folder_path=f"data/index/{name}", path_in_repo=name,
        commit_message=f"index {name}")
print("DONE indices ->",REPO,flush=True)
