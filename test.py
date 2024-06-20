import argparse

from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

from config.Config import *
from src.CIFAR100_Dataset import CIFAR100_Dataset
from src.Models import SimpleCNN, VisionTransformer
from src.Trainer import Trainer


def main(model_type):
    train_set, valid_set, test_set = CIFAR100_Dataset()
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers * n_gpus,
        collate_fn=default_collate,
        pin_memory=True
    )

    if model_type == "SimpleCNN":
        model = SimpleCNN(SimpleCNN_kwargs)
    elif model_type == "ViT":
        model = VisionTransformer(**ViT_kwargs)
    else:
        raise ValueError("model_type must be one of ['SimpleCNN', 'ViT']")

    model.load("models", f"{model_type}.ckpt")
    criterion = CrossEntropyLoss()

    trainer = Trainer(model, None, None, test_loader, criterion, None, None)
    trainer.test()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_type", type=str, required=True, choices=["SimpleCNN", "ViT"])
    args = parser.parse_args()

    main(args.model_type)
