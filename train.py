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


def train(pretrain_type="None", clear_logs=True):
    if clear_logs:
        import shutil
        shutil.rmtree(f"./logs/{pretrain_type}", ignore_errors=True)

    writer = SummaryWriter(log_dir=f"./logs/{pretrain_type}")
    steps = 0

    target_model = ResNet18(num_classes, pretrain_type).to(device)

    if pretrain_type == "None":
        logger.info("## Notice: The pretraining type of the model is none.")

        if n_gpus > 1:
            target_model = nn.DataParallel(target_model)

        dataloader = get_SL_loader()
        optimizer = Adam(target_model.parameters(), lr=lr)
        criterion = CrossEntropyLoss()

        logger.info("Training the complete model by supervised learning on CIFAR-100.")
        epochs = num_epochs_full
        for epoch in range(epochs):
            target_model.train()

            running_loss, correct, n = 0.0, 0, 0
            for i, (images, labels) in enumerate(dataloader):
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = target_model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                correct += (outputs.argmax(1) == labels).sum().item()
                n += images.size(0)
                
                steps += 1
                writer.add_scalar("Step Loss of Complete Model", loss.item(), steps)

            logger.info(f"Epoch {epoch + 1}/{epochs} | Loss: {(running_loss / n):.4f} | Acc: {(correct / n):.4f}")
            writer.add_scalar("Epoch Loss of Complete Model", running_loss / n, epoch)
            writer.add_scalar("Epoch Accuracy of Complete Model", correct / n, epoch)

    elif pretrain_type == "SL":
        logger.info("## Notice: The pretraining type of the model is supervised learning.")

        if n_gpus > 1:
            target_model = nn.DataParallel(target_model)

        dataloader = get_SL_loader()
        optimizer = Adam(target_model.parameters(), lr=lr)
        criterion = CrossEntropyLoss()

        logger.info("The feature net has been pretrained by supervised learning on ImageNet.")
        logger.info("Training the linear classifier by supervised learning on CIFAR-100.")
        epochs = num_epochs_full
        for epoch in range(epochs):
            target_model.train()

            running_loss, correct, n = 0.0, 0, 0
            for i, (images, labels) in enumerate(dataloader):
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = target_model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                correct += (outputs.argmax(1) == labels).sum().item()
                n += images.size(0)

                steps += 1
                writer.add_scalar("Step Loss of Classifier", loss.item(), steps)

            logger.info(f"Epoch {epoch + 1}/{epochs} | Loss: {(running_loss / n):.4f} | Acc: {(correct / n):.4f}")
            writer.add_scalar("Epoch Loss of Classifier", running_loss / n, epoch)
            writer.add_scalar("Epoch Accuracy of Classifier", correct / n, epoch)

    elif pretrain_type == "SSL":
        logger.info("## Notice: The pretraining type of the model is self-supervised learning.")

        SimCLR_model = SimCLR(
            base_model=target_model.feature_net,
            pj_head_dim=SimCLR_kwargs["pj_head_dim"]
        ).to(device)  # SimCLR的base_model与ResNet18的feature_net共享参数

        if n_gpus > 1:
            SimCLR_model = nn.DataParallel(SimCLR_model)

        SSL_dataloader = get_SimCLR_loader()
        SSL_optimizer = Adam(SimCLR_model.parameters(), lr=lr)

        logger.info("Pretraining the feature net by self-supervised learning on CIFAR-100.")
        epochs = num_epochs_pretrain
        for epoch in range(epochs):
            SimCLR_model.train()

            running_loss, n = 0.0, 0
            for i, (view1, view2, _) in enumerate(SSL_dataloader):
                view1, view2 = view1.to(device), view2.to(device)

                SSL_optimizer.zero_grad()
                z1, z2 = SimCLR_model(view1), SimCLR_model(view2)
                loss = SimCLR_Loss(z1, z2)
                loss.backward()
                SSL_optimizer.step()

                running_loss += loss.item()
                n += view1.size(0)

                steps += 1
                writer.add_scalar("Step Loss of Feature Net", loss.item(), steps)

            logger.info(f"Epoch {epoch + 1}/{epochs} | Loss: {(running_loss / n):.4f}")
            writer.add_scalar("Epoch Loss of Feature Net", running_loss / n, epoch)

        dataloader = get_SL_loader()
        optimizer = Adam(target_model.linear_classifier.parameters(), lr=lr)
        criterion = CrossEntropyLoss()

        if n_gpus > 1:
            target_model = nn.DataParallel(target_model)

        logger.info("Training the linear classifier by supervised learning on CIFAR-100.")
        epochs = num_epochs_full
        for epoch in range(epochs):
            target_model.train()

            running_loss, correct, n = 0.0, 0, 0
            for i, (images, labels) in enumerate(dataloader):
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = target_model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                correct += (outputs.argmax(1) == labels).sum().item()
                n += images.size(0)

                steps += 1
                writer.add_scalar("Step Loss of Classifier", loss.item(), steps)

            logger.info(f"Epoch {epoch + 1}/{epochs} | Loss: {(running_loss / n):.4f} | Acc: {(correct / n):.4f}")
            writer.add_scalar("Epoch Loss of Classifier", running_loss / n, epoch)
            writer.add_scalar("Epoch Accuracy of Classifier", correct / n, epoch)

    target_model.save("./models")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--pretrain_type", type=str, default="None", choices=["None", "SL", "SSL"])
    parser.add_argument("-c", "--clear_logs", type=bool, default=True, choices=[True, False])
    args = parser.parse_args()

    train(args.pretrain_type, args.clear_logs)
