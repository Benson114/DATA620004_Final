import argparse
import logging
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from src.Dataset import *
from src.Model import *

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def train(pretrain_type="None"):
    writer = SummaryWriter(log_dir=f"./logs/{pretrain_type}")

    if pretrain_type == "None":
        logger.info("Notice: The pretraining type of the model is none.")
        target_model = ResNet18(num_classes, pretrain_type).to(device)

        dataloader = get_SL_loader()
        optimizer = Adam(target_model.parameters(), lr=lr)
        criterion = CrossEntropyLoss()

        logger.info("Training the complete model by supervised learning.")
        epochs = num_epochs_full
        for epoch in range(epochs):
            target_model.train()

            running_loss, n = 0.0, 0
            for i, (images, labels) in enumerate(dataloader):
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = target_model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                n += images.size(0)

            logger.info(f"Epoch {epoch}/{epochs} | Loss: {loss.item()}")
            writer.add_scalar("Train Loss of Model", running_loss / n, epoch)

        target_model.save(f"./models")

    elif pretrain_type == "SL":
        logger.info("Notice: The pretraining type of the model is supervised learning.")
        target_model = ResNet18(num_classes, pretrain_type).to(device)

        dataloader = get_SL_loader()
        optimizer = Adam(target_model.parameters(), lr=lr)
        criterion = CrossEntropyLoss()

        logger.info("The feature net has been pretrained by supervised learning.")
        logger.info("Training the linear classifier by supervised learning.")
        epochs = num_epochs_full - num_epochs_pretrain
        for epoch in range(epochs):
            target_model.train()

            running_loss, n = 0.0, 0
            for i, (images, labels) in enumerate(dataloader):
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = target_model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                n += images.size(0)

            logger.info(f"Epoch {epoch}/{epochs} | Loss: {loss.item()}")
            writer.add_scalar("Train Loss of Classifier", running_loss / n, epoch)

        target_model.save(f"./models")

    elif pretrain_type == "SSL":
        logger.info("Notice: The pretraining type of the model is self-supervised learning.")
        target_model = ResNet18(num_classes, pretrain_type).to(device)
        SimCLR_model = SimCLR(
            base_model=target_model.feature_net,
            **SimCLR_kwargs
        ).to(device)  # SimCLR的base_model与ResNet18的feature_net共享参数

        SSL_dataloader = get_SimCLR_loader()
        SSL_optimizer = Adam(SimCLR_model.parameters(), lr=lr)

        logger.info("Pretraining the feature net by self-supervised learning.")
        epochs = num_epochs_pretrain
        for epoch in range(epochs):
            SimCLR_model.train()

            running_loss, n = 0.0, 0
            for i, (view1, view2, _) in enumerate(SSL_dataloader):
                view1, view2 = view1.to(device), view2.to(device)

                SSL_optimizer.zero_grad()
                z1, z2 = SimCLR_model(view1), SimCLR_model(view2)
                loss = SimCLR_model.loss(z1, z2)
                loss.backward()
                SSL_optimizer.step()

                running_loss += loss.item()
                n += view1.size(0)

            logger.info(f"Epoch {epoch}/{epochs} | Loss: {loss.item()}")
            writer.add_scalar("Train Loss of Feature Net", running_loss / n, epoch)

        dataloader = get_SL_loader()
        optimizer = Adam(target_model.linear_classifier.parameters(), lr=lr)
        criterion = CrossEntropyLoss()

        logger.info("Training the linear classifier by supervised learning.")
        epochs = num_epochs_full - num_epochs_pretrain
        for epoch in range(epochs):
            target_model.train()

            running_loss, n = 0.0, 0
            for i, (images, labels) in enumerate(dataloader):
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = target_model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                n += images.size(0)

            logger.info(f"Epoch {epoch}/{epochs} | Loss: {loss.item()}")
            writer.add_scalar("Train Loss of Classifier", running_loss / n, epoch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--pretrain_type", type=str, default="None", choices=["None", "SL", "SSL"])
    args = parser.parse_args()

    train(args.pretrain_type)
