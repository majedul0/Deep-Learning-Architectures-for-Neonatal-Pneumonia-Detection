"""
Flask dashboard for the pediatric/neonatal chest X-ray pneumonia classifiers.

Lets a clinician-facing user record basic patient info, upload a chest X-ray,
pick one of the six trained models (five fine-tuned transfer-learning backbones
+ the custom CNN baseline), and see the predicted label (NORMAL / PNEUMONIA)
with a Grad-CAM overlay and the model's benchmarked test-set performance.

Preprocessing here must match Untitled18.ipynb exactly (resize before CLAHE,
same CLAHE parameters) or predictions will silently degrade -- see CLAUDE.md.
Each saved .keras model already bakes its own normalization into the graph
(a Lambda(preprocess_input) for the transfer models, a Rescaling(1/255) layer
for the custom CNN), so the array handed to model.predict() here is the
resized + CLAHE'd RGB image in raw 0-255 range, nothing more.
"""

import csv
import os
import uuid
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, flash, redirect, render_template, request, url_for

import tensorflow as tf
from tensorflow.keras.applications import (
    densenet,
    efficientnet,
    mobilenet_v2,
    resnet50,
    vgg16,
)
from tensorflow.keras.models import Model, load_model

from hf_config import MODEL_FILES, MODEL_ROOT

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = (224, 224)
# Alphabetical order, matching tf.keras.utils.image_dataset_from_directory's
# class_names in Untitled19.ipynb / Untitled20.ipynb (NORMAL=0, PNEUMONIA=1).
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "tif", "tiff"}
MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10 MB

RESEARCH_TITLE = (
    "Deep Learning Architectures for Neonatal Pneumonia Detection in Chest "
    "X-ray Images: A Bias-Aware Comparison of Five CNN Backbones"
)

# The five backbones named in the research title (transfer learning); CustomCNN
# is the from-scratch baseline used for comparison, not one of "the five".
BACKBONE_MODELS = ["VGG16", "ResNet50", "DenseNet121", "EfficientNetB0", "MobileNetV2"]
BASELINE_MODELS = ["CustomCNN"]

# _final.keras is saved after phase 2 for the backbones (fine-tuned) and after
# the single training phase for CustomCNN -- this is the phase whose row in
# model_comparison.csv / sensitivity_specificity.csv matches the deployed weights.
MODEL_PHASE = {
    "VGG16": "finetuned",
    "ResNet50": "finetuned",
    "DenseNet121": "finetuned",
    "EfficientNetB0": "finetuned",
    "MobileNetV2": "finetuned",
    "CustomCNN": "single_phase",
}

# Each transfer-learning model bakes its own backbone's preprocess_input into a
# Lambda layer at save time (see build_model() in Untitled19.ipynb). Keras 3 only
# persists the bare function *name* ("preprocess_input") in the Lambda's config,
# not which backbone it came from, so it can't be resolved automatically on load
# -- we have to supply the correct one per model via custom_objects.
MODEL_PREPROCESS_FN = {
    "VGG16": vgg16.preprocess_input,
    "ResNet50": resnet50.preprocess_input,
    "DenseNet121": densenet.preprocess_input,
    "EfficientNetB0": efficientnet.preprocess_input,
    "MobileNetV2": mobilenet_v2.preprocess_input,
}

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-only-secret-key")
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

_model_cache = {}


def load_benchmarks():
    """Read the real per-model test-set metrics produced by Untitled19/20.ipynb."""
    accuracy_rows = {}
    comparison_csv = MODEL_ROOT / "model_comparison.csv"
    if comparison_csv.exists():
        with open(comparison_csv, newline="") as f:
            for row in csv.DictReader(f):
                accuracy_rows[(row["model"], row["phase"])] = row

    sens_spec_rows = {}
    sens_spec_csv = MODEL_ROOT / "sensitivity_specificity.csv"
    if sens_spec_csv.exists():
        with open(sens_spec_csv, newline="") as f:
            for row in csv.DictReader(f):
                sens_spec_rows[(row["model"], row["phase"])] = row

    complexity_rows = {}
    complexity_csv = MODEL_ROOT / "model_complexity.csv"
    if complexity_csv.exists():
        with open(complexity_csv, newline="") as f:
            for row in csv.DictReader(f):
                complexity_rows[row["model"]] = row

    benchmarks = {}
    for model_name, phase in MODEL_PHASE.items():
        acc_row = accuracy_rows.get((model_name, phase))
        ss_row = sens_spec_rows.get((model_name, phase))
        complexity_row = complexity_rows.get(model_name)
        if not acc_row:
            continue
        sensitivity = float(ss_row["sensitivity"]) if ss_row else None
        specificity = float(ss_row["specificity"]) if ss_row else None
        benchmarks[model_name] = {
            "phase": phase,
            "accuracy": float(acc_row["accuracy"]),
            "precision": float(acc_row["precision"]),
            "recall": float(acc_row["recall"]),
            "f1": float(acc_row["f1"]),
            "roc_auc": float(acc_row["roc_auc"]),
            "sensitivity": sensitivity,
            "specificity": specificity,
            "bias_gap": abs(sensitivity - specificity) if sensitivity is not None and specificity is not None else None,
            "total_params": int(complexity_row["total_params"]) if complexity_row else None,
        }
    return benchmarks


# Loaded once at startup; these are static evaluation artifacts from the notebooks,
# not something that changes while the server runs.
BENCHMARKS = load_benchmarks()


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def get_model(model_name):
    """Load a model on first use and cache it for subsequent requests."""
    if model_name in _model_cache:
        return _model_cache[model_name]

    model_path = MODEL_ROOT / MODEL_FILES[model_name]
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    preprocess_fn = MODEL_PREPROCESS_FN.get(model_name)
    custom_objects = {"preprocess_input": preprocess_fn} if preprocess_fn else None
    try:
        model = load_model(model_path, compile=False, custom_objects=custom_objects)
    except TypeError:
        # Keras 3's safe-mode blocks deserializing the Lambda(preprocess_fn)
        # layer baked into the transfer-learning models; these are our own
        # trusted training artifacts, so it's fine to disable safe mode here.
        model = load_model(
            model_path, compile=False, custom_objects=custom_objects, safe_mode=False
        )

    _model_cache[model_name] = model
    return model


def standardize_contrast(img_rgb):
    """CLAHE normalization -- must match Untitled18.ipynb exactly."""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)
    return cv2.cvtColor(gray_eq, cv2.COLOR_GRAY2RGB)


def load_and_preprocess(file_bytes):
    """Replicates Untitled18.ipynb's load_and_preprocess for an uploaded file."""
    arr = np.frombuffer(file_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image file")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_AREA)
    img = standardize_contrast(img)
    return img


def find_last_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    return None


def make_gradcam_heatmap(img_batch, model, last_conv_layer_name):
    grad_model = Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output],
    )
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_batch)
        loss = preds[:, 0]
    grads = tape.gradient(loss, conv_out)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_out = conv_out[0]
    heatmap = conv_out @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def save_gradcam_overlay(orig_img_rgb, heatmap, out_path):
    heatmap_resized = cv2.resize(heatmap, (orig_img_rgb.shape[1], orig_img_rgb.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    overlay = np.uint8(0.6 * orig_img_rgb + 0.4 * heatmap_color)
    cv2.imwrite(str(out_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))


@app.context_processor
def inject_globals():
    return {"research_title": RESEARCH_TITLE}


@app.route("/")
def index():
    return render_template(
        "index.html",
        backbone_models=BACKBONE_MODELS,
        baseline_models=BASELINE_MODELS,
        active_page="analyze",
    )


@app.route("/predict", methods=["POST"])
def predict():
    patient_name = (request.form.get("patient_name") or "").strip()
    patient_age = (request.form.get("patient_age") or "").strip()
    model_name = request.form.get("model_name")
    file = request.files.get("xray_image")

    if not patient_name:
        flash("Please enter the patient's name.")
        return redirect(url_for("index"))
    if not patient_age.isdigit() or not (0 <= int(patient_age) <= 120):
        flash("Please enter a valid age in days, months or years (0-120).")
        return redirect(url_for("index"))
    if model_name not in MODEL_FILES:
        flash("Please choose a valid model.")
        return redirect(url_for("index"))
    if not file or file.filename == "":
        flash("Please choose an X-ray image to upload.")
        return redirect(url_for("index"))
    if not allowed_file(file.filename):
        flash("Unsupported file type. Please upload a PNG, JPG, JPEG, BMP or TIFF image.")
        return redirect(url_for("index"))

    try:
        processed = load_and_preprocess(file.read())
    except ValueError:
        flash("That file could not be read as an image. Please try another.")
        return redirect(url_for("index"))

    try:
        model = get_model(model_name)
    except FileNotFoundError:
        flash(
            f"Model '{model_name}' is not available on this server "
            f"(expected at {MODEL_ROOT / MODEL_FILES[model_name]})."
        )
        return redirect(url_for("index"))

    img_batch = np.expand_dims(processed.astype(np.float32), axis=0)
    prob_pneumonia = float(model.predict(img_batch, verbose=0)[0][0])
    label = CLASS_NAMES[1] if prob_pneumonia >= 0.5 else CLASS_NAMES[0]
    confidence = prob_pneumonia if label == "PNEUMONIA" else 1.0 - prob_pneumonia

    request_id = uuid.uuid4().hex
    original_name = f"{request_id}_original.png"
    cv2.imwrite(str(UPLOAD_DIR / original_name), cv2.cvtColor(processed, cv2.COLOR_RGB2BGR))

    gradcam_name = None
    last_conv = find_last_conv_layer(model)
    if last_conv is not None:
        try:
            heatmap = make_gradcam_heatmap(img_batch, model, last_conv)
            gradcam_name = f"{request_id}_gradcam.png"
            save_gradcam_overlay(processed, heatmap, UPLOAD_DIR / gradcam_name)
        except Exception:
            gradcam_name = None  # Grad-CAM is a bonus visual; prediction still stands without it.

    return render_template(
        "result.html",
        active_page="analyze",
        patient_name=patient_name,
        patient_age=patient_age,
        analyzed_at=datetime.now().strftime("%d %b %Y, %H:%M"),
        model_name=model_name,
        label=label,
        confidence=round(confidence * 100, 1),
        benchmark=BENCHMARKS.get(model_name),
        original_image=url_for("static", filename=f"uploads/{original_name}"),
        gradcam_image=url_for("static", filename=f"uploads/{gradcam_name}") if gradcam_name else None,
    )


@app.route("/performance")
def performance():
    return render_template(
        "performance.html",
        active_page="performance",
        backbone_models=BACKBONE_MODELS,
        baseline_models=BASELINE_MODELS,
        benchmarks=BENCHMARKS,
    )


if __name__ == "__main__":
    # Port 5000 is Hyper-V/WinNAT-reserved on many Windows machines; default to 8000 instead.
    port = int(os.environ.get("PORT", 8000))
    app.run(host="127.0.0.1", port=port, debug=True, use_reloader=False)
