import os
import matplotlib.pyplot as plt
import numpy as np
import torchvision


class Visualization:

    def __init__(
        self,
        output_dir="outputs/figures"
    ):
        self.output_dir = output_dir

        os.makedirs(
            self.output_dir,
            exist_ok=True
        )


    def show_batch(
        self,
        images,
        filename="sample_images.png"
    ):

        grid = torchvision.utils.make_grid(
            images
        )

        img = grid.numpy()

        plt.figure(
            figsize=(8, 8)
        )

        plt.imshow(
            np.transpose(
                img,
                (1, 2, 0)
            )
        )

        plt.axis("off")

        path = os.path.join(
            self.output_dir,
            filename
        )

        plt.savefig(
            path,
            bbox_inches="tight"
        )

        plt.close()

        print(
            f"Saved: {path}"
        )


    def plot_training_curves(
        self,
        train_losses,
        test_accuracies
    ):

        # Loss curve

        plt.figure(
            figsize=(8, 5)
        )

        plt.plot(
            train_losses
        )

        plt.title(
            "Training Loss"
        )

        plt.xlabel(
            "Epoch"
        )

        plt.ylabel(
            "Loss"
        )

        loss_path = os.path.join(
            self.output_dir,
            "training_loss.png"
        )

        plt.savefig(
            loss_path,
            bbox_inches="tight"
        )

        plt.close()


        # Accuracy curve

        plt.figure(
            figsize=(8, 5)
        )

        plt.plot(
            test_accuracies
        )

        plt.title(
            "Test Accuracy"
        )

        plt.xlabel(
            "Epoch"
        )

        plt.ylabel(
            "Accuracy (%)"
        )

        acc_path = os.path.join(
            self.output_dir,
            "test_accuracy.png"
        )

        plt.savefig(
            acc_path,
            bbox_inches="tight"
        )

        plt.close()


        print(
            "Training curves saved!"
        )