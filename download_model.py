"""Download the ONNX model during deployment (used in Render's build command).

The model file is too big for GitHub's upload page, so it lives in a Hugging Face
model repository. The address is MODEL_URL in smartchef/settings.py.
"""
import os

import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "smartchef.settings")
django.setup()

from classifier.predictor import ensure_model_file  # noqa: E402  (needs Django set up first)

path = ensure_model_file()
print(f"Model ready: {path.name} ({path.stat().st_size / 1e6:.1f} MB)")
