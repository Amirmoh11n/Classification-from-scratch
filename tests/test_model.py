import torch

from models.cnn import CNNClassifier


def test_model_output():

    model = CNNClassifier()

    x = torch.randn(
        4,
        3,
        32,
        32
    )

    output = model(x)

    assert output.shape == (
        4,
        10
    )