import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


class CIFAR10DataModule:

    def __init__(
        self,
        batch_size,
        data_dir="./data"
    ):
        self.batch_size = batch_size
        self.data_dir = data_dir

    def get_loaders(self):

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5, 0.5, 0.5),
                (0.5, 0.5, 0.5)
            )
        ])

        train_dataset = torchvision.datasets.CIFAR10(
            root=self.data_dir,
            train=True,
            download=False,
            transform=transform
        )

        test_dataset = torchvision.datasets.CIFAR10(
            root=self.data_dir,
            train=False,
            download=False,
            transform=transform
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2
        )

        return train_loader, test_loader