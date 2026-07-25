import torch
import torch.nn as nn
import torch.optim as optim

from configs.config import Config
from datasets.cifar10_dataset import CIFAR10DataModule
from models.cnn import CNNClassifier
from trainers.trainer import Trainer

from utils.seed import set_seed
from utils.visualization import (
    plot_training_curves
)


def main():
    set_seed(Config.SEED)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Using device: {device}")

    data_module = CIFAR10DataModule(
        Config.BATCH_SIZE
    )

    train_loader, test_loader = (
        data_module.get_loaders()
    )

    model = CNNClassifier().to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=Config.LR
    )

    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device
    )

    train_losses = []
    test_accuracies = []

    best_acc = 0.0

    for epoch in range(Config.EPOCHS):

        loss = trainer.train_epoch(
            train_loader
        )

        acc = trainer.evaluate(
            test_loader
        )

        train_losses.append(loss)
        test_accuracies.append(acc)

        print(
            f"Epoch [{epoch+1}/{Config.EPOCHS}] "
            f"Loss: {loss:.4f} "
            f"Test Acc: {acc:.2f}%"
        )

        if acc > best_acc:

            best_acc = acc

            torch.save(
                model.state_dict(),
                Config.MODEL_SAVE_PATH
            )

            print(
                f"Best model saved "
                f"(Acc={best_acc:.2f}%)"
            )

    print(
        f"\nTraining Finished!"
        f"\nBest Accuracy: {best_acc:.2f}%"
    )

    plot_training_curves(
        train_losses,
        test_accuracies
    )


if __name__ == "__main__":
    main()