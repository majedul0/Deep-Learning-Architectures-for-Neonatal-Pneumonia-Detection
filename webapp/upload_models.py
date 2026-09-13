"""One-time helper: upload the 6 trained model weight files to a Hugging Face
Hub model repo, so a deployment (e.g. a Hugging Face Space) can fetch them at
build time instead of needing them committed to git.

Run this locally, not as part of the deployed app:

    pip install huggingface_hub
    huggingface-cli login          # paste a token with WRITE access, from
                                    # https://huggingface.co/settings/tokens
    python webapp/upload_models.py

Set the HF_MODEL_REPO env var first if you want a repo id other than the
default in hf_config.py, e.g. (PowerShell):

    $env:HF_MODEL_REPO = "your-username/your-repo-name"
    python webapp/upload_models.py
"""

from huggingface_hub import HfApi

from hf_config import HF_MODEL_REPO, MODEL_FILES, MODEL_ROOT


def main():
    api = HfApi()
    api.create_repo(repo_id=HF_MODEL_REPO, repo_type="model", exist_ok=True)

    uploaded, skipped = [], []
    for model_name, rel_path in MODEL_FILES.items():
        local_path = MODEL_ROOT / rel_path
        if not local_path.exists():
            print(f"[skip] {model_name}: not found at {local_path}")
            skipped.append(model_name)
            continue
        print(f"[upload] {model_name} -> {HF_MODEL_REPO}/{rel_path}")
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=rel_path,
            repo_id=HF_MODEL_REPO,
            repo_type="model",
        )
        uploaded.append(model_name)

    print(f"\nUploaded: {', '.join(uploaded) or 'none'}")
    if skipped:
        print(f"Skipped (file not found locally): {', '.join(skipped)}")
    print(f"Repo: https://huggingface.co/{HF_MODEL_REPO}")


if __name__ == "__main__":
    main()
