"""Convert the trained PyTorch model to ONNX, a lighter format for serving.

Run this once on your laptop (inside the project's .venv):

    pip install onnx onnxscript onnxruntime
    python export_onnx.py

It reads  model/smartchef_resnet18.pth  and  model/class_names.json,
writes    model/smartchef.onnx,
and checks that the ONNX model gives the same answers as the PyTorch one.
"""
import inspect
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
from torchvision import models

MODEL_DIR = Path(__file__).resolve().parent / "model"

with open(MODEL_DIR / "class_names.json") as f:
    class_names = json.load(f)

# Rebuild the same network as in the notebook and load the trained weights
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(torch.load(MODEL_DIR / "smartchef_resnet18.pth", map_location="cpu"))
model.eval()

example = torch.randn(1, 3, 224, 224)
onnx_path = MODEL_DIR / "smartchef.onnx"

# Different PyTorch versions accept different options, so only pass the ones this version knows
options = {
    "input_names": ["image"],
    "output_names": ["logits"],
    "dynamic_axes": {"image": {0: "batch"}, "logits": {0: "batch"}},
    "opset_version": 17,
}
accepted = inspect.signature(torch.onnx.export).parameters
if "dynamo" in accepted:
    options["dynamo"] = False          # the classic exporter: one self-contained file
if "external_data" in accepted:
    options["external_data"] = False   # keep the weights inside the .onnx file

torch.onnx.export(model, example, str(onnx_path), **options)
print(f"Saved {onnx_path.name} ({onnx_path.stat().st_size / 1e6:.1f} MB)")

# Check: both models should give (almost) the same scores
with torch.no_grad():
    expected = model(example).numpy()
session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
actual = session.run(None, {"image": example.numpy()})[0]

diff = float(np.abs(expected - actual).max())
print(f"Largest difference between PyTorch and ONNX scores: {diff:.6f}")
print("Check passed" if diff < 1e-3 else "Check FAILED: the outputs differ")
