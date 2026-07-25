import torch

class Trainer:

    def __init__(
        self,
        model,
        criterion,
        optimizer,
        device
    ):

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

    def train_epoch(self, loader):

        self.model.train()

        total_loss = 0

        for images, labels in loader:

            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(images)

            loss = self.criterion(outputs, labels)

            loss.backward()

            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(loader)

    def evaluate(self, loader):

        self.model.eval()

        correct = 0
        total = 0

        with torch.no_grad():

            for images, labels in loader:

                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)

                _, preds = torch.max(outputs, 1)

                total += labels.size(0)

                correct += (preds == labels).sum().item()

        return 100 * correct / total