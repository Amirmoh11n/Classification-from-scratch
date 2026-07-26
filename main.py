import os

import torch
import torch.nn as nn

from configs.config import Config
from datasets.cifar10_dataset import CIFAR10DataModule
from models.model import CNNClassifier
from trainers.trainer import Trainer

from utils.seed import set_seed
from utils.visualization import Visualization


def main():

    set_seed(Config.SEED)

    os.makedirs(
        "checkpoints",
        exist_ok=True
    )


    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Using device: {device}"
    )


    data_module = CIFAR10DataModule(
        Config.BATCH_SIZE
    )


    train_loader, test_loader = (
        data_module.get_loaders()
    )


    visualizer = Visualization()


    model = CNNClassifier(
        num_classes=Config.NUM_CLASSES
    ).to(device)


    criterion = nn.CrossEntropyLoss(
        label_smoothing=0.1
    )


    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=Config.LR,
        momentum=0.9,
        weight_decay=5e-4
    )


    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=Config.EPOCHS
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


        scheduler.step()


        train_losses.append(loss)

        test_accuracies.append(acc)


        current_lr = optimizer.param_groups[0]["lr"]


        print(
            f"Epoch [{epoch+1}/{Config.EPOCHS}] "
            f"Loss: {loss:.4f} "
            f"Acc: {acc:.2f}% "
            f"LR: {current_lr:.6f}"
        )


        if acc > best_acc:

            best_acc = acc


            torch.save(
                model.state_dict(),
                Config.MODEL_SAVE_PATH
            )


            print(
                f"✓ Best model saved "
                f"(Acc={best_acc:.2f}%)"
            )


    print(
        "\nTraining Finished!"
    )

    print(
        f"Best Accuracy: {best_acc:.2f}%"
    )


    visualizer.plot_training_curves(
        train_losses,
        test_accuracies
    )


if __name__ == "__main__":
    main()