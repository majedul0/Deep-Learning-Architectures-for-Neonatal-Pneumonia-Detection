# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

A research project on pediatric chest X-ray pneumonia classification, plus a design doc for a
lightweight web demo of the winning model. It is **not a conventional codebase**: there is no git repo,
no package manager, no build/lint/test tooling, and no application source code yet. The actual work
lives in three Google Colab notebooks (run against Google Drive paths under `/content/drive/MyDrive/...`,
not anything local) and a folder of pre-generated training artifacts.

- `Untitled18.ipynb` — **dataset builder**. Combines a "primary" clean pediatric X-ray dataset with a
  small supplement of raw/phone-photo pneumonia images, balances classes by augmenting the primary
  images up to a target count per class, then splits into `train/val/test` (80/10/10) under
  `Final_Dataset_Split`.
- `Untitled19.ipynb` — **training + reporting notebook** for the five transfer-learning backbones
  (VGG16, ResNet50, DenseNet121, EfficientNetB0, MobileNetV2). Cells 1–9 train; cells `R1`–`R9` (run
  in a separate Colab session after training) reload results from Drive and render the comparison
  report (per-model curves/metrics, Grad-CAM galleries, cross-model comparison table/charts, dataset
  overview, sensitivity/specificity table, model complexity table, environment info).
- `Untitled20.ipynb` — **Custom CNN baseline**, trained from scratch (no ImageNet weights, single
  phase, no fine-tuning split) for comparison against the transfer-learning models.
- `deployment_architecture.md` — design doc for a single-process Flask web app that would serve the
  MobileNetV2 model as a `.tflite` file with Grad-CAM overlays, targeting Render's free tier (512MB
  RAM). Describes an intended deployment; the Flask app itself (`app.py`, `templates/`, etc.) does not
  exist in this repo — only the design doc does.
- `Results-20260913T045822Z-1-001/Results/` (and the identical `Results-...zip`) — the output artifacts
  from a completed run of `Untitled19.ipynb` / `Untitled20.ipynb`: per-model, per-phase folders each
  containing `*.keras`/`best.weights.h5`, `metrics.json`, `classification_report.csv`, and PNG curves
  (accuracy/loss/ROC/PR/confusion matrix), plus a `gradcam/` subfolder of overlay images, and
  top-level `model_comparison.csv`, `sensitivity_specificity.csv`, `model_complexity.csv`,
  `environment_info.json`.

## Working with the notebooks

There is no local run/build command — these are Colab notebooks that mount Google Drive
(`/content/drive/MyDrive/Chest X-ray Pneumonia/...`) and are executed cell-by-cell in a Colab runtime.
When editing notebook code in this repo, edit the `.ipynb` JSON directly (or via a notebook-aware tool);
there's no way to execute or lint them locally without recreating that Drive layout and a Colab/GPU
environment. Treat path constants at the top of each notebook (`DRIVE_DATA_DIR`, `RESULTS_DIR`,
`OUTPUT_ROOT`, etc.) as the source of truth for the expected directory layout rather than assuming
anything about this repo's own file structure.

`Untitled19.ipynb`'s training loop (`train_all_models`, cell 13) is resumable: it checks
`is_done(phase_dir)` (presence of `metrics.json`) per model/phase and skips already-completed
work, so re-running the notebook after a partial/interrupted run continues rather than retraining
everything.

## Pipeline architecture

**1. Dataset construction (`Untitled18.ipynb`)** — every image goes through
`load_and_preprocess`: BGR→RGB, optional `autocrop_black_border` (used only for the raw phone-photo
source, to strip camera-capture borders), resize to 224×224, then `standardize_contrast` (CLAHE,
`clipLimit=2.0`, `tileGridSize=(8,8)`, applied on the grayscale image before converting back to RGB).
Class balancing works by capping the raw phone-photo supplement at `RAW_SUPPLEMENT_MAX_RATIO` (15%) of
the target count, then filling any remaining shortfall via `augment_image` (random one-of: flip,
rotate, zoom, brightness, shift, Gaussian noise) applied to the primary images. A sanity check trains a
throwaway logistic-regression classifier to confirm raw vs. clean images aren't trivially separable
after preprocessing (which would mean the model could "cheat" by detecting image source instead of
pathology). **Any inference pipeline (including the planned web app) must replicate this exact
preprocessing sequence — resize before CLAHE, same CLAHE parameters — or predictions will silently
degrade.**

**2. Transfer-learning models (`Untitled19.ipynb`)** — all five backbones share one
`build_model(app_fn, preprocess_fn)`: a Keras `Input` → a `Lambda(preprocess_fn)` (the backbone's own
ImageNet `preprocess_input`, baked into the graph) → the frozen `app_fn(include_top=False,
weights='imagenet')` base → `GlobalAveragePooling2D` → `Dropout(0.3)` → `Dense(128, relu)` →
`Dropout(0.2)` → `Dense(1, sigmoid)`. Training is two-phase per model, and both phases' artifacts are
kept (not just the final one) for the comparison report:
  - **Phase 1 (`phase1_no_finetune`)**: base frozen, head trained at `lr=1e-3`.
  - **Phase 2 (`phase2_finetuned`)**: `unfreeze_top_layers` unfreezes the last 30 layers of the base
    (keeping any `BatchNormalization` layers in that range frozen) and retrains the whole model at
    `lr=1e-5`.

  Both phases use `EarlyStopping(patience=5, monitor='val_loss', restore_best_weights=True)`,
  `ReduceLROnPlateau`, and checkpoint the best weights only. Class imbalance is handled via
  `compute_class_weight` (inverse-frequency) rather than resampling at train time. Grad-CAM
  (`find_last_conv_layer` + custom NumPy/OpenCV implementation) is generated per model on a few
  correct and misclassified test examples for explainability.

**3. Custom CNN baseline (`Untitled20.ipynb`)** — a from-scratch 4-block Conv2D/BatchNorm/MaxPool CNN
(32→64→128→256 filters) with augmentation and rescaling built into the model graph itself (not the
`tf.data` pipeline), trained single-phase with no ImageNet weights. Its much lower reported accuracy
(~0.56 vs. ~0.95–0.98 for the transfer-learning models — see `model_comparison.csv`) is the expected
baseline result establishing that transfer learning is necessary at this dataset size, not a bug to fix.

## Key results (from `Results/model_comparison.csv`, `sensitivity_specificity.csv`, `model_complexity.csv`)

- **VGG16 (fine-tuned)** is the best-performing model overall (accuracy 0.985, ROC-AUC 0.999) but is
  also the most compute-heavy of the transfer-learning models (~14.8M total params).
- **MobileNetV2** is the smallest transfer-learning model (~2.4M params) with competitive accuracy
  (0.971 fine-tuned) — this is why it, not VGG16, was chosen for the resource-constrained web deployment
  described in `deployment_architecture.md`. Don't "correct" this to VGG16 without re-reading that
  doc's accuracy-vs-efficiency reasoning (Section 11).
- Fine-tuning (phase 2) doesn't always beat phase 1 no-finetune (e.g. DenseNet121 and EfficientNetB0
  score lower fine-tuned) — this is a real, already-recorded result, not evidence of a training bug.

## If asked to build the web app

`deployment_architecture.md` is a design doc, not a description of existing code — there is no
`app.py`, `templates/`, or `requirements.txt` in this repo yet. If implementing it: use the
`MobileNetV2_final.keras` in `Results/MobileNetV2/`, convert to `.tflite` per Section 4 of the doc, and
reproduce the exact preprocessing sequence from `Untitled18.ipynb` (Section 5 of the doc / "Pipeline
architecture" step 1 above) rather than reinventing it.
