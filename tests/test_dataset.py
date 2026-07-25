import torch

from datasets.cifar10_dataset import CIFAR10DataModule


def test_dataloader():

    data = CIFAR10DataModule(
        batch_size=16
    )

    train_loader, test_loader = (
        data.get_loaders()
    )

    images, labels = next(
        iter(train_loader)
    )

    assert images.shape == (
        16, 3, 32, 32
    )

    assert labels.shape == (
        16,
    )

    assert labels.max() < 10