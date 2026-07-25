# CIFAR-10 Image Classification with CNN

## Overview

This project implements an image classification system for the CIFAR-10 dataset using a Convolutional Neural Network (CNN) built with PyTorch.

The project follows Object-Oriented Programming (OOP) principles and a modular architecture to ensure maintainability, scalability, and code readability.

The model is trained to classify images into one of ten categories:

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
├── models/
│   ├── __init__.py
│   └── cnn.py
│
├── datasets/
│   ├── __init__.py
│   └── cifar10_dataset.py
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

## Features

* CIFAR-10 image classification
* Convolutional Neural Network (CNN)
* Modular OOP design
* Reproducible experiments with fixed random seeds
* Automatic model checkpoint saving
* Training loss monitoring
* Test accuracy evaluation
* Training curve visualization
* GPU acceleration support (CUDA)

---

## Technologies Used

* Python 3.x
* PyTorch
* Torchvision
* NumPy
* Matplotlib

---

## CNN Architecture

The network consists of:

### Feature Extractor

1. Conv2D (3 → 32)

2. ReLU

3. MaxPooling

4. Conv2D (32 → 64)

5. ReLU

6. MaxPooling

7. Conv2D (64 → 128)

8. ReLU

9. MaxPooling

### Classifier

1. Flatten
2. Linear (2048 → 256)
3. ReLU
4. Linear (256 → 10)

---

## Installation

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

## Running the Project

Start training with:

```bash
python main.py
```

The CIFAR-10 dataset will be downloaded automatically on the first run.

---

## Training Output

During training, the console displays:

```text
Epoch [1/20] Loss: 1.4253 Test Acc: 49.32%
Epoch [2/20] Loss: 1.0871 Test Acc: 58.14%
...
```

The best model is automatically saved in:

```text
checkpoints/cnn_best.pth
```

---

## Reproducibility

To ensure consistent experimental results, a fixed random seed is used:

```python
set_seed(42)
```

This controls randomness in:

* Python
* NumPy
* PyTorch
* CUDA

---

## Results Visualization

After training, the project generates:

* Training Loss Curve
* Test Accuracy Curve

These plots help analyze model convergence and performance.

---

## Future Improvements

Possible enhancements include:

* Data augmentation
* Batch normalization
* Dropout regularization
* Learning rate scheduling
* Early stopping
* TensorBoard integration
* Confusion matrix visualization
* Transfer learning with ResNet or EfficientNet

---

## Educational Purpose

This project was developed as a learning exercise in:

* Deep Learning
* Computer Vision
* Convolutional Neural Networks
* PyTorch
* Object-Oriented Design
* Machine Learning Experiment Management

---

## License

This project is available for educational and research purposes.
