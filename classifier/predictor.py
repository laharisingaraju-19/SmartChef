"""Turns a photo into an ingredient and a smoothie recipe.

The model is the ResNet18 trained in the SmartChef Kaggle notebook (transfer
learning on Fruits-360, 163 classes), exported to ONNX with export_onnx.py.
ONNX Runtime needs far less memory than PyTorch, which lets the app run on a
small free server. Preprocessing matches the notebook's validation transforms.
"""
import json
import re
import urllib.request
from functools import lru_cache

import numpy as np
import onnxruntime as ort
from django.conf import settings
from PIL import Image

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def ensure_model_file():
    """Download the ONNX model from Hugging Face if it isn't in the model folder yet."""
    path = settings.MODEL_DIR / "smartchef.onnx"
    if not path.exists():
        print(f"Downloading model from {settings.MODEL_URL}")
        urllib.request.urlretrieve(settings.MODEL_URL, path)
    return path


@lru_cache(maxsize=1)
def load_model():
    """Open the ONNX model and class names once, then reuse them for every request."""
    with open(settings.MODEL_DIR / "class_names.json") as f:
        class_names = json.load(f)

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1  # the free server has a small share of one CPU
    session = ort.InferenceSession(
        str(ensure_model_file()), options, providers=["CPUExecutionProvider"]
    )
    return session, class_names


@lru_cache(maxsize=1)
def load_recipes():
    with open(settings.MODEL_DIR / "recipes.json") as f:
        return json.load(f)


def preprocess(image):
    """Resize to 224x224, scale to 0-1, normalize with ImageNet stats, and reorder to (1, 3, 224, 224).

    Same steps as transforms.Resize((224, 224)) + ToTensor() + Normalize(...) in the notebook.
    """
    image = image.convert("RGB").resize((224, 224), Image.BILINEAR)
    pixels = np.asarray(image, dtype=np.float32) / 255.0
    pixels = (pixels - MEAN) / STD
    return pixels.transpose(2, 0, 1)[np.newaxis, :].astype(np.float32)


def softmax(scores):
    exp = np.exp(scores - scores.max())
    return exp / exp.sum()


def clean(label):
    """'apple_red_yellow_1' -> 'apple red yellow' (drops the variety number)."""
    text = label.lower().replace("_", " ")
    text = re.sub(r"\s*\d+$", "", text)
    return text.strip()


def find_recipe(label):
    """First recipe that uses this ingredient. Single-ingredient recipes are listed first."""
    name = clean(label)
    for recipe in load_recipes():
        for item in recipe["ingredients"]:
            item = clean(item)
            if name == item or name.startswith(item + " ") or item.startswith(name + " "):
                return recipe
    return None


def predict(image, top_k=3):
    """Return the top guesses as (name, percent) pairs, plus the matching recipe or None."""
    session, class_names = load_model()
    scores = session.run(None, {"image": preprocess(image)})[0][0]
    probs = softmax(scores)

    top = np.argsort(probs)[::-1][:top_k]
    guesses = [(clean(class_names[i]).title(), round(float(probs[i]) * 100, 1)) for i in top]
    return guesses, find_recipe(class_names[top[0]])
