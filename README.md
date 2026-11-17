# SmartChef

A Django web app that identifies a fruit or vegetable from a photo and suggests a smoothie recipe.

You upload a photo, a fine-tuned **ResNet18** model classifies it into one of 163 produce classes, the page shows the top three guesses with confidence scores, and the top guess is matched to a smoothie recipe with fun facts.

## How it works

| Part | What it does |
|---|---|
| Training | ResNet18 pre-trained on ImageNet, fine-tuned with transfer learning on Fruits-360 (163 classes) in PyTorch. Notebook: `SmartChef.ipynb` and [on Kaggle](https://www.kaggle.com/code/laharisingaraju19/smartchef-ingredient-classification-with-resnet18) |
| `export_onnx.py` | Converts the trained PyTorch model to ONNX and checks both give the same scores |
| `classifier/predictor.py` | Runs the ONNX model with ONNX Runtime, using the same resize and normalization as training, and returns the top 3 classes |
| `classifier/views.py` | Validates the upload, fixes phone photo rotation, runs the prediction and renders the result. Photos are processed in memory and never saved |
| `model/recipes.json` | Smoothie recipes keyed by ingredient, matched to the predicted class |

**Why ONNX:** serving with ONNX Runtime instead of PyTorch cuts memory use enough to run on a free 512 MB server.

## Run it locally

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python manage.py runserver
```

Then open http://127.0.0.1:8000. On first use the app downloads `model/smartchef.onnx` from Hugging Face.

## Deploy (Render)

- Build command: `pip install -r requirements.txt && python download_model.py && python manage.py collectstatic --noinput`
- Start command: `gunicorn smartchef.wsgi:application`
- Environment variables: `DJANGO_DEBUG=0`, `DJANGO_SECRET_KEY=<a long random string>`

## Tech stack

Python · Django · PyTorch · ONNX Runtime · ResNet18 · Transfer learning · Pillow
