import torch


class Config:

    BATCH_SIZE = 64
    EPOCHS = 20
    LR = 0.001

    NUM_CLASSES = 10

    MODEL_SAVE_PATH = (
        "checkpoints/cnn_best.pth"
    )

    DEVICE = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    SEED = 42