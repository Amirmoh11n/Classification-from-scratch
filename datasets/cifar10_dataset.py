import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

class CIFAR10DataModule:

    def __init__(self, batch_size):
        self.batch_size = batch_size

    def get_loaders(self):

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5, 0.5, 0.5),
                (0.5, 0.5, 0.5)
            )
        ])

        train_dataset = torchvision.datasets.CIFAR10(
            root="./data",
            train=True,
            download=True,
            transform=transform
        )

        test_dataset = torchvision.datasets.CIFAR10(
            root="./data",
            train=False,
            download=True,
            transform=transform
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False
        )

        return train_loader, test_loader