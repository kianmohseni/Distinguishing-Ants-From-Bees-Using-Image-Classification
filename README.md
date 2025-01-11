# Project Overview

This project builds a machine learning classifier to distinguish between images of ants and bees. It leverages a pre-trained neural network, ResNet50, for feature extraction and applies machine learning models like Logistic Regression to classify the images.

The workflow includes:

1. Image Preprocessing: Images are normalized and resized to meet ResNet50's input requirements.
2. Feature Extraction: ResNet50 generates a 2048-dimensional representation for each image.
3. Classification: A logistic regression classifier is trained on the extracted features to perform the final classification.

# Key Features
1. Data Handling:
- Utilizes PyTorch's datasets and transforms to load and preprocess data.
- Training and validation datasets are resized to 224x224 and normalized.
2. Transfer Learning:
- ResNet50, a pre-trained deep learning model, is used for feature extraction.
- The network's parameters are frozen to focus on high-level feature representation.
3. Classification Models:
- Logistic Regression: Achieves ~80% accuracy on the test set.
- K-Nearest Neighbors (k-NN): Explored with varying values of k.

# Libraries and Tools
- PyTorch: For data preprocessing and feature extraction using ResNet50.
- Scikit-learn: For training and evaluating classifiers like Logistic Regression and k-NN.
- Matplotlib: For data visualization and inspection.

# Dataset
- Source: A collection of labeled images of ants and bees.
- Structure:
  - Training Set: 244 images.
  - Validation Set: 153 images.
- Classes: Two labels, ants and bees.

# Model Performance
## Logistic Regression:
- Accuracy: 80.39%

## K-Nearest Neighbors:
- Test Accuracy:
  - k=1: 69.28%
  - k=3: 71.24%
  - k=5: 69.28%

# Future Enhancements
- Experiment with other pre-trained models like VGG16 or EfficientNet.
- Implement data augmentation for better generalization.
- Explore fine-tuning ResNet50 for improved performance.
