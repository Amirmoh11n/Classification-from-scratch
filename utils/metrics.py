import torch

def accuracy(outputs, labels):

    _, preds = torch.max(outputs, 1)

    correct = (preds == labels).sum().item()

    return correct / labels.size(0)