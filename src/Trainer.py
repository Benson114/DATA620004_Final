import time
import torch.nn as nn

from config.Config import *


class Trainer:
    def __init__(self, model, train_loader, valid_loader, test_loader, criterion, optimizer, writer):
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.writer = writer

        if n_gpus > 1:
            self.model = nn.DataParallel(self.model)

    def train_epoch(self, epoch):
        self.model.train()
        self.model.to(device)

        running_loss, correct, n = 0.0, 0, 0

        for i, (inputs, labels) in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            labels1, labels2, lam = labels
            outputs = outputs.to(device)
            labels1 = labels1.to(device)
            labels2 = labels2.to(device)
            loss = lam * self.criterion(outputs, labels1) + (1 - lam) * self.criterion(outputs, labels2)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            correct += (
                    lam * (outputs.argmax(1) == labels1).float() + (1 - lam) * (outputs.argmax(1) == labels2).float()
            ).sum().item()
            n += labels1.size(0)

        self.writer.add_scalar("Train Loss", running_loss / n, epoch)
        self.writer.add_scalar("Train Accuracy", correct / n, epoch)

        return running_loss / n, correct / n

    def valid_epoch(self, epoch, is_test=False):
        self.model.eval()
        self.model.to(device)

        running_loss, correct, n = 0.0, 0, 0

        with torch.no_grad():
            if is_test:
                loader = self.test_loader
            else:
                loader = self.valid_loader

            for i, (inputs, labels) in enumerate(loader):
                outputs = self.model(inputs)
                labels1, labels2, lam = labels
                outputs = outputs.to(device)
                labels1 = labels1.to(device)
                labels2 = labels2.to(device)
                loss = lam * self.criterion(outputs, labels1) + (1 - lam) * self.criterion(outputs, labels2)

                running_loss += loss.item()
                correct += (
                        lam * (outputs.argmax(1) == labels1).float() + (1 - lam) * (
                            outputs.argmax(1) == labels2).float()
                ).sum().item()
                n += labels1.size(0)

        if n == 0:
            return None, None

        if not is_test:
            self.writer.add_scalar("Valid Loss", running_loss / n, epoch)
            self.writer.add_scalar("Valid Accuracy", correct / n, epoch)

        return running_loss / n, correct / n

    def train(self):
        for epoch in range(num_epochs):
            start_time = time.time()
            train_loss, train_acc = self.train_epoch(epoch)
            valid_loss, valid_acc = self.valid_epoch(epoch, is_test=False)
            end_time = time.time()
            if valid_loss is None:
                continue
            print(f"Epoch {epoch + 1}/{num_epochs} | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                  f"Valid Loss: {valid_loss:.4f} | Valid Acc: {valid_acc:.4f} | "
                  f"Time: {end_time - start_time:.2f}s")

    def test(self):
        test_loss, test_acc = self.valid_epoch(0, is_test=True)
        print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")
