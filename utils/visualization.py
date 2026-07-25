import matplotlib.pyplot as plt
import numpy as np
import torchvision


def show_batch(images):

    grid = torchvision.utils.make_grid(images)

    img = grid.numpy()

    plt.figure(figsize=(8, 8))

    plt.imshow(
        np.transpose(img, (1, 2, 0))
    )

    plt.axis("off")

    plt.show()


def plot_training_curves(
    train_losses,
    test_accuracies
):

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)

    plt.plot(train_losses)

    plt.title("Train Loss")

    plt.xlabel("Epoch")

    plt.ylabel("Loss")

    plt.subplot(1, 2, 2)

    plt.plot(test_accuracies)

    plt.title("Test Accuracy")

    plt.xlabel("Epoch")

    plt.ylabel("Accuracy (%)")

    plt.tight_layout()

    plt.show()