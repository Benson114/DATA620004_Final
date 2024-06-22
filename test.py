import argparse
import logging
from torch.nn import CrossEntropyLoss

from src.Dataset import *
from src.Model import *

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def test(pretrain_type="None"):
    target_model = ResNet18(num_classes, pretrain_type).to(device)
    target_model.load("./models")
    target_model.eval()

    if n_gpus > 1:
        target_model = nn.DataParallel(target_model)

    dataloader = get_test_loader()
    criterion = CrossEntropyLoss()

    logger.info(f"Testing the model pretrained by {pretrain_type}.")
    running_loss, correct, n = 0.0, 0, 0
    with torch.no_grad():
        for i, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)

            outputs = target_model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
            n += images.size(0)

    logger.info(f"Loss: {(running_loss / n):.4f} | Acc: {(correct / n):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--pretrain_type", type=str, default="None", choices=["None", "SSL", "SL"])
    args = parser.parse_args()

    test(args.pretrain_type)
