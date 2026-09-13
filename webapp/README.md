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

Then open http://127.0.0.1:5000 in a browser.

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
