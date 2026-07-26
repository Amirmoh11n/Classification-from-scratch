# CIFAR-10 Image Classification with Deep CNN (PyTorch)

## Overview

This project implements an image classification system on the **CIFAR-10 dataset** using a custom deep Convolutional Neural Network built with **PyTorch**.

The project follows a clean **Object-Oriented Programming (OOP)** architecture with a modular design for training, evaluation, visualization, and experimentation.

The model is trained from scratch to classify images into 10 categories:

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

# Project Structure

```text
cifar10-cnn-classification/
│
├── configs/
│   └── config.py
│
├── data/
│   └── cifar-10-batches-py/
│
├── datasets/
│   ├── cifar10_dataset.py
│   └── __init__.py
│
├── models/
│   ├── model.py
│   └── __init__.py
│
├── trainers/
│   ├── trainer.py
│   └── __init__.py
│
├── utils/
│   ├── seed.py
│   ├── visualization.py
│   ├── metrics.py
│   └── __init__.py
│
├── tests/
│   ├── test_dataset.py
│   ├── test_model.py
│   └── test_trainer.py
│
├── checkpoints/
│
├── outputs/
│   └── figures/
│
├── main.py
├── requirements.txt
└── README.md
```

---

# Features

* CIFAR-10 image classification
* Custom deep CNN architecture
* ResNet-style residual blocks
* Training from scratch using PyTorch
* Object-Oriented project design
* Data augmentation pipeline
* GPU acceleration support
* Reproducible experiments
* Automatic checkpoint saving
* Training visualization
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

The model uses a custom **ResNet-inspired architecture** designed specifically for CIFAR-10.

## Network Overview

```text
Input Image
(3 x 32 x 32)

        |
        v

Initial Convolution

        |
        v

Residual Block × 2
64 Channels

        |
        v

Residual Block × 2
128 Channels

        |
        v

Residual Block × 2
256 Channels

        |
        v

Residual Block × 2
512 Channels

        |
        v

Adaptive Average Pooling

        |
        v

Fully Connected Layer

        |
        v

10 Class Output
```

---

# Training Strategy

The training pipeline uses modern deep learning techniques:

## Optimizer

Stochastic Gradient Descent:

```python
SGD(
    lr=0.1,
    momentum=0.9,
    weight_decay=5e-4
)
```

## Learning Rate Scheduler

Cosine Annealing:

```python
CosineAnnealingLR(
    T_max=epochs
)
```

## Loss Function

Cross Entropy with label smoothing:

```python
CrossEntropyLoss(
    label_smoothing=0.1
)
```

---

# Data Augmentation

To improve generalization, the training dataset uses:

* Random Crop
* Horizontal Flip
* Color Jitter
* CIFAR-10 normalization

Example:

```python
RandomCrop(
    32,
    padding=4
)

RandomHorizontalFlip()
```

The validation/test dataset only uses normalization to provide fair evaluation.

---

# Installation

Clone the repository:

```bash
git clone <repository-url>

cd cifar10-cnn-classification
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

# Dataset

This project uses the CIFAR-10 dataset.

Dataset structure:

```text
data/
└── cifar-10-batches-py/
    ├── data_batch_1
    ├── data_batch_2
    ├── data_batch_3
    ├── data_batch_4
    ├── data_batch_5
    ├── test_batch
    └── batches.meta
```

The dataset is loaded locally and does not require downloading during execution.

---

# Training

Run:

```bash
python main.py
```

The training process will:

1. Load CIFAR-10 dataset
2. Apply augmentation
3. Train the CNN model
4. Evaluate accuracy
5. Save the best checkpoint
6. Generate training curves

---

# Checkpoints

The best model is automatically saved:

```text
checkpoints/cnn_best.pth
```

---

# Visualization

Training results are saved automatically:

```text
outputs/
└── figures/
    ├── training_loss.png
    └── test_accuracy.png
```

Generated plots include:

* Training loss curve
* Test accuracy curve

---

# Testing

The project includes automated tests using PyTest.

Run:

```bash
pytest tests/
```

Verbose mode:

```bash
pytest -v
```

Tests cover:

### Dataset

* Image dimensions
* Label format
* Class range validation

### Model

* Forward pass
* Output dimensions
* 10-class prediction

### Trainer

* Training component initialization

---

# Reproducibility

Random seeds are controlled using:

```python
set_seed(42)
```

This ensures reproducible experiments across:

* Python
* NumPy
* PyTorch
* CUDA

---

# Results

Current model performance:

| Model                      | Accuracy |
| -------------------------- | -------: |
| Basic CNN                  |     ~76% |
| Deep CNN + Residual Blocks |    ~85%+ |

Further improvements can be achieved with:

* MixUp
* CutMix
* WideResNet
* Advanced schedulers
* Longer training

---

# Future Improvements

Possible extensions:

* Confusion matrix visualization
* Precision / Recall / F1 metrics
* TensorBoard integration
* Mixed precision training
* MixUp and CutMix augmentation
* WideResNet implementation
* Model deployment with FastAPI

---

# Learning Objectives

This project demonstrates:

* Computer Vision fundamentals
* CNN architectures
* Residual learning
* PyTorch training workflows
* Deep Learning optimization techniques
* ML project engineering practices

---

# License

This project is released for educational and research purposes.
