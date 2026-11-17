"""Loads the trained ResNet18 once and turns a photo into an ingredient and a recipe.

The model was trained in the SmartChef Kaggle notebook (transfer learning on
Fruits-360, 163 classes). Image preprocessing here matches the notebook's
validation transforms, so predictions match what the notebook would give.
"""
import json
import re
from functools import lru_cache

import torch
import torch.nn as nn
from django.conf import settings
from torchvision import models, transforms

PREPROCESS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


@lru_cache(maxsize=1)
def load_model():
    """Build ResNet18 with the output layer sized to the class list and load the trained weights (only once)."""
    with open(settings.MODEL_DIR / "class_names.json") as f:
        class_names = json.load(f)

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    state = torch.load(settings.MODEL_DIR / "smartchef_resnet18.pth", map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model, class_names


@lru_cache(maxsize=1)
def load_recipes():
    with open(settings.MODEL_DIR / "recipes.json") as f:
        return json.load(f)


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
    model, class_names = load_model()
    batch = PREPROCESS(image.convert("RGB")).unsqueeze(0)

    with torch.no_grad():
        probs = torch.softmax(model(batch), dim=1)[0]

    top = torch.topk(probs, top_k)
    guesses = [
        (clean(class_names[i]).title(), round(float(p) * 100, 1))
        for p, i in zip(top.values, top.indices)
    ]
    best_label = class_names[top.indices[0]]
    return guesses, find_recipe(best_label)
