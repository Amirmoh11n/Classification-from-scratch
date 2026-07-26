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

        train_transform = transforms.Compose([

            transforms.RandomCrop(
                32,
                padding=4
            ),

            transforms.RandomHorizontalFlip(),

            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2
            ),

            transforms.ToTensor(),

            transforms.Normalize(
                mean=(
                    0.4914,
                    0.4822,
                    0.4465
                ),

                std=(
                    0.2470,
                    0.2435,
                    0.2616
                )
            )
        ])


        test_transform = transforms.Compose([

            transforms.ToTensor(),

            transforms.Normalize(
                mean=(
                    0.4914,
                    0.4822,
                    0.4465
                ),

                std=(
                    0.2470,
                    0.2435,
                    0.2616
                )
            )
        ])


        train_dataset = torchvision.datasets.CIFAR10(
            root=self.data_dir,
            train=True,
            download=False,
            transform=train_transform
        )


        test_dataset = torchvision.datasets.CIFAR10(
            root=self.data_dir,
            train=False,
            download=False,
            transform=test_transform
        )


        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )


        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )


        return train_loader, test_loader