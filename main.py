import argparse

import torch.optim as optim
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.tensorboard import SummaryWriter

from config.Config import *
from src.CIFAR100_Dataset import CIFAR100_Dataset
from src.CIFAR100_DataAugmentation import collate_fn_cutmix
from src.Models import SimpleCNN, VisionTransformer
from src.Trainer import Trainer


def main(model_type, clear_logs=True):
    if clear_logs:
        import shutil
        shutil.rmtree(f"./logs/{model_type}", ignore_errors=True)

    writer = SummaryWriter(log_dir=f"./logs/{model_type}")

    train_set, valid_set, test_set = CIFAR100_Dataset()
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers * n_gpus,
        collate_fn=collate_fn_cutmix,
        pin_memory=True
    )
    valid_loader = DataLoader(
        valid_set,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers * n_gpus,
        collate_fn=default_collate,
        pin_memory=True
    )
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

    criterion = CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), **optimizer_kwargs)

    trainer = Trainer(model, train_loader, valid_loader, test_loader, criterion, optimizer, writer)
    trainer.train()
    trainer.test()

    model.save("models", f"{model_type}.ckpt")

    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_type", type=str, required=True, choices=["SimpleCNN", "ViT"])
    parser.add_argument("-c", "--clear_logs", type=bool, default=True)
    args = parser.parse_args()

    main(args.model_type, args.clear_logs)
