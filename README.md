# SmartChef

A Django web app that identifies a fruit or vegetable from a photo and suggests a smoothie recipe.

You upload a photo, a fine-tuned **ResNet18** model classifies it into one of 163 produce classes, the page shows the top three guesses with confidence scores, and the top guess is matched to a smoothie recipe with fun facts.

## How it works

| Part | What it does |
|---|---|
| Model | ResNet18 pre-trained on ImageNet, fine-tuned with transfer learning on the Fruits-360 dataset (163 classes) in PyTorch. Training notebook: [SmartChef on Kaggle](https://www.kaggle.com/code/laharisingaraju19/smartchef-ingredient-classification-with-resnet18) |
| `classifier/predictor.py` | Loads the trained weights once, applies the same resize and normalization used in training, and returns the top 3 classes with probabilities |
| `classifier/views.py` | Validates the upload, fixes phone photo rotation, runs the prediction and renders the result. Photos are processed in memory and never saved |
| `model/recipes.json` | Smoothie recipes keyed by ingredient, matched to the predicted class |

## Run it locally

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python manage.py runserver
```

Then open http://127.0.0.1:8000.

The `model/` folder needs three files:

- `smartchef_resnet18.pth`: trained weights, saved from the Kaggle notebook
- `class_names.json`: the 163 class names in training order
- `recipes.json`: included

## Tech stack

Python · Django · PyTorch · torchvision · ResNet18 · Transfer learning · Pillow
