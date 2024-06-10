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
            inputs, labels = inputs.to(device), labels.to(device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
            n += labels.size(0)

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
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item()
                correct += (outputs.argmax(1) == labels).sum().item()
                n += labels.size(0)

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
            print(f"Epoch {epoch + 1}/{num_epochs} | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                  f"Valid Loss: {valid_loss:.4f} | Valid Acc: {valid_acc:.4f} | "
                  f"Time: {end_time - start_time:.2f}s")

    def test(self):
        test_loss, test_acc = self.valid_epoch(0, is_test=True)
        print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")
