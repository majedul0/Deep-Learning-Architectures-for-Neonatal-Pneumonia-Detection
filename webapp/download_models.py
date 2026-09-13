"""Fetch the trained model weights from the Hugging Face Hub model repo.

Run as a Docker build step (see ../Dockerfile) so the image is self-contained
and the deployed app never needs network access to serve a prediction. Safe
to re-run: existing files are left alone.

Requires the weights to already be uploaded once via upload_models.py.
"""

import sys

from huggingface_hub import hf_hub_download

from hf_config import HF_MODEL_REPO, MODEL_FILES, MODEL_ROOT


def main():
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    missing = []

    for model_name, rel_path in MODEL_FILES.items():
        target = MODEL_ROOT / rel_path
        if target.exists():
            print(f"[skip] {model_name}: already present at {target}")
            continue
        print(f"[fetch] {model_name} <- {HF_MODEL_REPO}/{rel_path}")
        try:
            hf_hub_download(
                repo_id=HF_MODEL_REPO,
                filename=rel_path,
                local_dir=MODEL_ROOT,
            )
        except Exception as exc:
            print(f"[error] {model_name}: {exc}")
            missing.append(model_name)

    if missing:
        print(f"\nFailed to fetch: {', '.join(missing)}")
        print(f"Check that {HF_MODEL_REPO} exists and contains these files")
        print("(upload them first with upload_models.py), or set HF_MODEL_REPO")
        print("to the correct repo id.")
        sys.exit(1)


if __name__ == "__main__":
    main()
