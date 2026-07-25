# CIFAR-10 Image Classification with CNN

## Overview

This project implements an image classification system on the CIFAR-10 dataset using a custom Convolutional Neural Network (CNN) built with PyTorch.

The project is designed with a modular **Object-Oriented Programming (OOP)** architecture to provide clean code organization, scalability, and easy experimentation.

The model learns to classify images into 10 different categories:

* Airplane
* Automobile
* Bird
* Cat
* Deer
* Dog
* Frog
* Horse
* Ship
* Truck

---

## Project Structure

```text
cifar10_cnn_classifier/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── configs/
│   └── config.py
│
├── datasets/
│   ├── __init__.py
│   └── cifar10_dataset.py
│
├── models/
│   ├── __init__.py
│   └── cnn.py
│
├── trainers/
│   ├── __init__.py
│   └── trainer.py
│
├── utils/
│   ├── __init__.py
│   ├── metrics.py
│   ├── visualization.py
│   └── seed.py
│
├── tests/
│   ├── __init__.py
│   ├── test_dataset.py
│   ├── test_model.py
│   └── test_trainer.py
│
├── checkpoints/
│
├── outputs/
│   ├── figures/
│   └── logs/
│
├── main.py
├── requirements.txt
└── README.md
```

---

# Features

* CIFAR-10 image classification
* Custom CNN architecture using PyTorch
* Object-Oriented Programming design
* Modular project structure
* GPU acceleration support
* Reproducible training with fixed random seeds
* Automatic best model checkpoint saving
* Training loss and accuracy visualization
* Unit testing with PyTest

---

# Technologies

* Python 3.x
* PyTorch
* Torchvision
* NumPy
* Matplotlib
* PyTest

---

# Model Architecture

The CNN consists of two main parts:

## Feature Extractor

```
Conv2D (3 → 32)
ReLU
MaxPooling

Conv2D (32 → 64)
ReLU
MaxPooling

Conv2D (64 → 128)
ReLU
MaxPooling
```

## Classifier

```
Flatten

Linear (2048 → 256)
ReLU

Linear (256 → 10)
```

The final layer produces probabilities for the 10 CIFAR-10 classes.

---

# Installation

Clone the repository:

```bash
git clone <repository-url>

cd cifar10_cnn_classifier
```

Create a virtual environment:

```bash
python -m venv venv

source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Training

Run the training pipeline:

```bash
python main.py
```

The dataset will be automatically downloaded on the first execution.

During training:

```text
Epoch [1/20] Loss: 1.42 Test Acc: 49.32%
Epoch [2/20] Loss: 1.08 Test Acc: 58.14%
...
```

The best model is saved automatically:

```
checkpoints/cnn_best.pth
```

---

# Testing

This project includes automated tests using **PyTest**.

Run all tests:

```bash
pytest tests/
```

Run with detailed output:

```bash
pytest -v
```

The tests verify:

## Dataset

* Correct image shape
* Correct label format
* Valid CIFAR-10 class range

## Model

* Correct CNN forward pass
* Correct output dimension
* 10-class prediction support

## Trainer

* Correct initialization
* Optimizer and loss function integration

---

# Reproducibility

To make experiments repeatable, random seeds are controlled using:

```python
set_seed(42)
```

This controls randomness in:

* Python
* NumPy
* PyTorch
* CUDA

---

# Visualization

The project generates training analysis:

* Training Loss Curve
* Test Accuracy Curve

Generated figures can be stored in:

```
outputs/figures/
```

---

# Future Improvements

Possible extensions:

* Data augmentation
* Batch Normalization
* Dropout layers
* Learning rate scheduler
* Early stopping
* TensorBoard logging
* Confusion matrix
* Classification report
* Transfer learning with ResNet / EfficientNet
* Model deployment with FastAPI

---

# Learning Goals

This project was created to practice:

* Deep Learning fundamentals
* Computer Vision
* CNN architectures
* PyTorch workflow
* Model training pipelines
* Machine Learning project organization
* Testing and reproducibility

---

# License

This project is intended for educational and research purposes.
