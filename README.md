# SmartChef

An ingredient classifier that identifies fruits and vegetables from photos and suggests a smoothie recipe for what it sees.

## Overview

SmartChef uses transfer learning to classify images of ingredients. A ResNet18 network pretrained on ImageNet is adapted to 84 fruit and vegetable classes from the Fruits-360 dataset. The predicted ingredient is then matched to a recipe, with instructions, extra ingredients and nutrition facts.

This repository contains the model training notebook. The trained model was also integrated into a Django web application for real-time ingredient classification.

## Results

| Metric | Value |
|---|---|
| Classes | 84 |
| Training images | About 28,000 |
| Test images | 14,046 |
| Training accuracy | 99.98% |
| Test accuracy | 100% |
| Training time | 25 minutes for 10 epochs on a Kaggle GPU |

The notebook includes the loss and accuracy curves, a normalized confusion matrix, and a per-class classification report with precision, recall and F1 score.

## Approach

- **Model:** ResNet18 pretrained on ImageNet. The convolutional layers are frozen and the final fully connected layer is replaced with a new 84-class output layer, which is the only part trained.
- **Preprocessing:** Images are resized to 224 x 224 and normalized with the ImageNet mean and standard deviation. Training images are also randomly flipped horizontally.
- **Training:** Cross-entropy loss, Adam optimizer with a learning rate of 0.001, batch size of 32, and a step scheduler that cuts the learning rate by 10x every 5 epochs. The weights with the best test accuracy are kept.
- **Evaluation:** Accuracy, confusion matrix and classification report on the Fruits-360 test split.
- **Recipe matching:** Each predicted label is looked up in a dictionary of smoothie recipes. Labels are matched flexibly, so variants such as "Avocado Green 1" and "Avocado Black 1" both map to the avocado recipe.

## Example

The last section of the notebook runs the model on photos from outside the dataset and prints the matched recipe:

```
Recipe for Avocado Green 1: Avocado Smoothie
Instructions: Blend 1/2 avocado with 1/2 cup of milk.
Additional Ingredients: milk
```

## Limitations

Fruits-360 images show a single item on a plain background, which makes the test set much easier than real kitchen photos. The 100% test accuracy reflects that, and accuracy on cluttered, real-world images will be lower. Fine-tuning the deeper layers and training on more varied photos are the natural next steps.

## Repository contents

| File | Description |
|---|---|
| `smartchef-ingredient-classification-with-resnet18.ipynb` | Data loading, model setup, training, evaluation and recipe matching |

## Running the notebook

The notebook was built on Kaggle, and the easiest way to run it is there:

[SmartChef on Kaggle](https://www.kaggle.com/code/laharisingaraju19/smartchef-ingredient-classification-with-resnet18)

To run it locally:

1. Install the dependencies:

   ```bash
   pip install torch torchvision numpy pillow matplotlib seaborn scikit-learn tqdm jupyter
   ```

2. Download the [Fruits-360 dataset](https://www.kaggle.com/datasets/moltean/fruits) and point `data_dir` in the notebook at the `fruits-360-original-size` folder.
3. To try your own photos, put them in a folder and point `custom_folder` at it.
4. Open the notebook and run all cells:

   ```bash
   jupyter notebook smartchef-ingredient-classification-with-resnet18.ipynb
   ```

A GPU is recommended for training.

## Tech stack

Python, PyTorch, torchvision, ResNet18, scikit-learn, seaborn, Django

## Author

Soundarya Lahari Singaraju

- GitHub: [laharisingaraju-19](https://github.com/laharisingaraju-19)
- LinkedIn: [soundaryalaharisingaraju](https://www.linkedin.com/in/soundaryalaharisingaraju)
