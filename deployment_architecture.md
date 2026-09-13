# Neonatal/Pediatric Pneumonia Detection — Web Deployment Architecture

This document describes the architecture, technology stack, and deployment details for the web-based demo of the pneumonia detection model developed in this project.

---

## 1. Overview

The deployment is a single, self-contained web application: one Python service serves both the user-facing HTML/CSS pages and the model inference logic. There is no separate frontend/backend split (e.g., no PHP + Python API pair) — everything runs inside one Flask app, which keeps the architecture simple enough to fit comfortably within a free hosting tier.

The core design goal is **low resource usage**, since the target hosting platform (Render's free tier) provides only 512 MB RAM and 0.1 CPU per service. Every technology choice below was made with that constraint in mind.

---

## 2. Architecture Diagram

```
                        ┌─────────────────────────────┐
                        │         User's Browser        │
                        │   (uploads a chest X-ray)     │
                        └───────────────┬───────────────┘
                                        │ HTTP POST (image)
                                        ▼
                        ┌─────────────────────────────┐
                        │      Flask Web Application     │
                        │        (single service)        │
                        │                                 │
                        │  1. Receive uploaded image       │
                        │  2. Preprocess (resize, CLAHE)   │
                        │  3. Run TFLite inference          │
                        │  4. Generate Grad-CAM overlay      │
                        │  5. Render HTML result page         │
                        └───────────────┬───────────────┘
                                        │
                                        ▼
                        ┌─────────────────────────────┐
                        │   mobilenetv2_pneumonia.tflite │
                        │      (bundled model file)      │
                        └─────────────────────────────┘
```

No database, no external API calls, and no separate backend service — the entire request/response cycle happens inside one process.

---

## 3. Technology Stack

| Layer | Technology | Why |
|---|---|---|
| Web framework | **Flask** (Python) | Lightweight, minimal memory footprint, serves both HTML pages and handles file uploads/inference in one app |
| Frontend | **HTML + CSS** (Jinja2 templates) | No JavaScript framework needed for a simple upload-and-view-result flow; keeps client-side footprint at zero |
| Model inference | **TensorFlow Lite** (`tflite-runtime`) | Full TensorFlow's runtime alone can use 200–400 MB RAM; TFLite is a lightweight, optimized runtime built for exactly this kind of memory-constrained environment |
| Deployed model | **MobileNetV2** (fine-tuned), converted to `.tflite` | Smallest and most efficient of the five architectures evaluated in this project (~3.5M parameters); the only one realistically small enough for a 512 MB free-tier instance |
| Image preprocessing | **OpenCV (headless)** | Must exactly replicate the training pipeline: resize to 224×224, CLAHE contrast normalization |
| Explainability | **Grad-CAM** (custom implementation, NumPy + OpenCV) | Reused from the training notebook's evaluation pipeline, adapted to run on a single uploaded image rather than a batch |
| Hosting | **Render** (free tier) | 512 MB RAM / 0.1 CPU per service, no credit card required, deploys directly from a Git repository |
| Version control / deploy trigger | **GitHub** | Render auto-deploys on every push to the connected repository |

---

## 4. Why TensorFlow Lite Instead of Full TensorFlow

This is the single most important architectural decision in this deployment, so it's worth documenting explicitly:

- The model used throughout training (`tensorflow.keras`) is the full TensorFlow library, which is unnecessarily heavy for serving a single already-trained model.
- TensorFlow Lite is a separate, much smaller runtime designed specifically for inference on resource-constrained devices (originally built for mobile/embedded use).
- Converting the trained `.keras` model to `.tflite` format, with default optimizations applied, reduces both the model file size and the runtime memory needed to load and run it — the difference between "fits in 512 MB" and "likely crashes or fails to deploy."

Conversion is a one-time step performed after training, not part of the live web app:

```python
import tensorflow as tf

model = tf.keras.models.load_model("MobileNetV2_final.keras")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open("mobilenetv2_pneumonia.tflite", "wb") as f:
    f.write(tflite_model)
```

---

## 5. Preprocessing Pipeline (Must Match Training Exactly)

Predictions are only reliable if the uploaded image is processed identically to how training images were processed. The web app's preprocessing step replicates the training pipeline precisely:

1. Read uploaded image, convert to RGB
2. Resize to 224 × 224 (same as training)
3. Apply CLAHE contrast normalization (same parameters as the dataset-builder notebook)
4. Normalize pixel values as expected by MobileNetV2's `preprocess_input`

Any deviation from this sequence (different resize method, skipped CLAHE step, wrong normalization range) will degrade prediction quality even though the model itself is unchanged.

---

## 6. Request Lifecycle

1. User visits the site, sees an upload form (HTML/CSS, served by Flask)
2. User selects and submits a chest X-ray image
3. Flask receives the file via a POST request
4. Image is preprocessed (Section 5)
5. The TFLite interpreter runs inference, producing a probability (0–1)
6. Prediction is thresholded at 0.5 → "Pneumonia" or "Normal" label
7. Grad-CAM is computed on the same image to produce a heatmap overlay
8. Flask renders a results page showing: original image, predicted label, confidence score, and the Grad-CAM overlay
9. Render's free-tier service spins down automatically after 15 minutes of inactivity; the next visit triggers a ~30–60 second cold start while the service wakes up

---

## 7. Project File Structure

```
pneumonia-webapp/
├── app.py                          # Flask application (routes, inference logic)
├── requirements.txt                # Python dependencies (see Section 8)
├── model/
│   └── mobilenetv2_pneumonia.tflite
├── static/
│   └── style.css
├── templates/
│   ├── index.html                  # Upload form
│   └── result.html                 # Prediction + Grad-CAM display
└── render.yaml                     # Render deployment configuration (optional)
```

---

## 8. Dependencies (`requirements.txt`)

```
flask
tflite-runtime
opencv-python-headless
numpy
pillow
```

Note: `opencv-python-headless` (not the standard `opencv-python`) is used deliberately — the headless variant excludes GUI dependencies that aren't needed on a server and aren't available in Render's environment, further reducing installed package size.

---

## 9. Hosting Constraints and Their Implications

| Constraint | Value | Implication for this project |
|---|---|---|
| RAM | 512 MB | Rules out deploying VGG16 (best-accuracy model) directly with full TensorFlow; MobileNetV2 + TFLite is the only combination from this project confirmed to fit comfortably |
| CPU | 0.1 vCPU | Inference on a single image takes longer than on a full GPU-backed training environment, but is acceptable for a demo (not production-scale traffic) |
| Inactivity spin-down | 15 minutes | First visitor after idle time experiences a 30–60 second delay while the service restarts; not an issue for a defense demo, would matter for real-world continuous use |
| Monthly free compute | 750 instance-hours | Sufficient for a low-traffic academic demo; would need a paid tier for sustained production use |

---

## 10. Limitations of This Deployment

- **Only MobileNetV2 is deployed**, not all five architectures evaluated in the project. This is a deliberate, documented trade-off driven by free-tier resource limits, not a reflection of MobileNetV2 being the most accurate model (VGG16 remains the best-performing model reported in Chapter 4 of the written report).
- **No authentication, rate-limiting, or input validation beyond basic file-type checks** — this is a demo/prototype, not a production clinical tool, consistent with the "decision-support, not diagnostic replacement" framing established in the project report.
- **Cold-start delay** on the free tier means the first prediction after a period of inactivity will be noticeably slower than subsequent ones.

---

## 11. Relationship to the Main Research Project

This deployment is a proof-of-concept extension of the trained models described in the written report (Chapters 3–4) and is not itself a contribution being evaluated as part of the core research — it exists to demonstrate that the trained pipeline can be made accessible outside a Colab notebook. The choice of MobileNetV2 for deployment directly reflects the accuracy-versus-efficiency trade-off discussed in Section 4.3 of the report: it is the architecture best suited to "resource-limited clinical settings," and this deployment is a literal demonstration of that conclusion.
