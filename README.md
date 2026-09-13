# Deep Learning Architectures for Neonatal Pneumonia Detection in Chest X-ray Images: A Bias-Aware Comparison of Five CNN Backbones

Pediatric chest X-ray pneumonia classification research, comparing five transfer-learning
CNN backbones against a from-scratch custom CNN baseline, plus a Flask web dashboard for
interactively testing the trained models.

## Contents

- **`Untitled18.ipynb`** — dataset builder. Combines a clean pediatric X-ray dataset with a
  small supplement of raw/phone-photo pneumonia images, balances classes, and splits into
  train/val/test (80/10/10).
- **`Untitled19.ipynb`** — trains and evaluates five transfer-learning backbones (VGG16,
  ResNet50, DenseNet121, EfficientNetB0, MobileNetV2), each in two phases (frozen head, then
  fine-tuned), and generates the full comparison report (metrics, ROC/PR curves, Grad-CAM,
  sensitivity/specificity, model complexity).
- **`Untitled20.ipynb`** — custom CNN baseline trained from scratch (no ImageNet weights), for
  comparison against the transfer-learning models.
- **`deployment_architecture.md`** — design doc for a lightweight single-model (MobileNetV2 +
  TFLite) deployment targeting Render's free tier.
- **`Results-20260913T045822Z-1-001/Results/`** — training run artifacts: per-model metrics,
  classification reports, curves, Grad-CAM galleries, and the cross-model comparison CSVs.
  Trained model weights (`*.keras`, `*.h5`) are not tracked in this repository (see below);
  everything else — metrics, reports, charts — is.
- **`webapp/`** — a Flask dashboard for interactively testing the trained models: record
  patient info, upload a chest X-ray, pick a model, and get a NORMAL/PNEUMONIA prediction with
  a Grad-CAM overlay and the model's benchmarked test-set performance. See
  [`webapp/README.md`](webapp/README.md) for setup and run instructions.

## Key results

See `Results-20260913T045822Z-1-001/Results/model_comparison.csv` and
`sensitivity_specificity.csv` for full numbers. Headline result: VGG16 (fine-tuned) is the
best-performing model overall (accuracy 0.985, ROC-AUC 0.999), while MobileNetV2 offers
competitive accuracy (0.971 fine-tuned) at a fraction of the parameter count — the basis for
choosing it in the resource-constrained deployment design.

## Model weights

Trained model weights are excluded from this repository via `.gitignore` — several of the
fine-tuned checkpoints (e.g. VGG16, ResNet50) exceed GitHub's 100MB per-file limit, and
together they run into gigabytes. Regenerate them by running `Untitled19.ipynb` /
`Untitled20.ipynb` against the dataset produced by `Untitled18.ipynb`, or obtain them
separately if you need the exact trained weights used to produce these results.
