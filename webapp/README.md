# Pneumonia X-ray Classifier — Web Dashboard

A single-process Flask app for the pediatric chest X-ray pneumonia project. Upload an
X-ray, pick one of the 6 trained models, and see the prediction (NORMAL / PNEUMONIA)
with a Grad-CAM overlay.

This is a broader dashboard than the one described in `../deployment_architecture.md`:
that doc specs a resource-constrained single-model (MobileNetV2 + TFLite) deployment
for Render's free tier. This app instead loads the `.keras` files directly and lets you
switch between all 6 models (VGG16, ResNet50, DenseNet121, EfficientNetB0, MobileNetV2,
CustomCNN) — useful for local comparison, not for the 512 MB free-tier target.

## Setup

**Python version matters here.** TensorFlow does not yet publish wheels for very new
Python releases (e.g. 3.13/3.14) — use **Python 3.10, 3.11, or 3.12** in a dedicated
virtual environment, even if a newer Python is your system default.

```powershell
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Model files

The app expects the trained models at:

```
../Results-20260913T045822Z-1-001/Results/<ModelName>/<ModelName>_final.keras
```

relative to this folder (i.e. the `Results/` directory already in this repo). To point
at a different location, set the `MODEL_ROOT` environment variable to the folder that
directly contains `VGG16/`, `ResNet50/`, etc.

Models are loaded lazily (on first request for that model) and cached in memory, so
startup is fast but the first prediction with a given model will take a few seconds.

## Run

```powershell
python app.py
```

Then open http://127.0.0.1:8000 in a browser. (Port 8000, not Flask's usual 5000 —
5000 is reserved by Windows' Hyper-V/WinNAT port range on many machines; override with
the `PORT` env var if you need a different one.)

## Deploying to Hugging Face Spaces

The model weight files (`*.keras`) are gitignored from this repo — several exceed
GitHub's 100MB file limit — so they aren't available wherever this repo gets cloned.
Deployment works around that by hosting the weights in a separate Hugging Face Hub
**model repo** and having the Space's Docker build fetch them at build time (see
`../Dockerfile`, `download_models.py`, `hf_config.py`). This keeps the deployed image
self-contained (no download needed while actually serving requests) and requires no
Git LFS or external file hosting for the main project repo.

1. **Create a free Hugging Face account** at https://huggingface.co if you don't have one.

2. **Upload your trained weights once** (from this project's Python environment, with
   `huggingface_hub` installed — already in `requirements.txt`):

   ```powershell
   huggingface-cli login
   # paste a token with WRITE access, from https://huggingface.co/settings/tokens
   python upload_models.py
   ```

   This creates a public model repo (default id in `hf_config.py`:
   `majedul0/neonatal-pneumonia-cnn-weights`; override with the `HF_MODEL_REPO` env var
   for a different name) and uploads all 6 `_final.keras` files to it.

3. **Create the Space**: go to https://huggingface.co/new-space, pick **Docker** as the
   SDK, name it, choose public or private visibility, and create it. Hugging Face gives
   you a git URL like `https://huggingface.co/spaces/<username>/<space-name>`.

4. **Push this repo to the Space** as a second git remote alongside GitHub:

   ```bash
   git remote add space https://huggingface.co/spaces/<username>/<space-name>
   git push space main
   ```

   When prompted for credentials, use your Hugging Face username and the same access
   token as the password.

5. The Space detects `sdk: docker` in the repo's root `README.md` and builds
   `../Dockerfile` automatically — this installs dependencies, then runs
   `download_models.py` to bake in the weights from your model repo. The first build
   takes a few minutes (downloading TensorFlow + ~360MB of weights); after that, visit
   the Space's URL to use the live dashboard.

6. To update the live Space later, just push to it again: `git push space main`.

## Notes

- Preprocessing (resize to 224×224, then CLAHE with `clipLimit=2.0`,
  `tileGridSize=(8,8)`) exactly replicates `Untitled18.ipynb`'s `load_and_preprocess`.
  Each model's own normalization (`preprocess_input` for the transfer models,
  `Rescaling(1/255)` for the custom CNN) is baked into the saved `.keras` graph, so no
  extra scaling is applied here.
- Class order is `["NORMAL", "PNEUMONIA"]`, matching the alphabetical class order
  `image_dataset_from_directory` used during training.
- Grad-CAM is best-effort: if the heatmap can't be computed for a given model, the
  prediction is still shown without the overlay.
- No authentication, rate-limiting, or clinical validation — this is a research/demo
  tool, not a diagnostic device.
