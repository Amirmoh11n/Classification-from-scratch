import torch

import torch.nn as nn
import torch.optim as optim

from models.cnn import CNNClassifier
from trainers.trainer import Trainer


def test_trainer_creation():

    model = CNNClassifier()

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=0.001
    )


    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device="cpu"
    )


    assert trainer.model is model

    assert trainer.criterion is criterion

    assert trainer.optimizer is optimizer