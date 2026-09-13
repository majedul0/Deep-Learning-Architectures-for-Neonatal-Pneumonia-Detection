"""Shared config for locating the trained model weights and, for deployment,
fetching them from a Hugging Face Hub model repo.

Kept free of heavy imports (no TensorFlow) so download_models.py can run as an
early Docker build step before installing/importing TensorFlow at all.
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Results-20260913T045822Z-1-001/Results lives one level up from webapp/.
# Override with the MODEL_ROOT env var if the Results folder is moved elsewhere.
MODEL_ROOT = Path(
    os.environ.get(
        "MODEL_ROOT",
        BASE_DIR.parent / "Results-20260913T045822Z-1-001" / "Results",
    )
)

# Hugging Face Hub model repo the deployed weights are uploaded to / fetched
# from. Override with the HF_MODEL_REPO env var if you use a different repo id.
HF_MODEL_REPO = os.environ.get("HF_MODEL_REPO", "majedul0/neonatal-pneumonia-cnn-weights")

MODEL_FILES = {
    "VGG16": "VGG16/VGG16_final.keras",
    "ResNet50": "ResNet50/ResNet50_final.keras",
    "DenseNet121": "DenseNet121/DenseNet121_final.keras",
    "EfficientNetB0": "EfficientNetB0/EfficientNetB0_final.keras",
    "MobileNetV2": "MobileNetV2/MobileNetV2_final.keras",
    "CustomCNN": "CustomCNN/CustomCNN_final.keras",
}
