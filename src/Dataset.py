import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.Config import *

train_set = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transforms.ToTensor())
test_set = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transforms.ToTensor())


class SimCLRDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, idx):
        image, label = self.dataset[idx]

        if self.transform:
            view1 = self.transform(image)
            view2 = self.transform(image)
        else:
            view1 = image
            view2 = image

        return view1, view2, label

    def __len__(self):
        return len(self.dataset)


def get_SimCLR_transform(color_jitter_p, gray_scale_p, normalize):
    transform = transforms.Compose([
        transforms.RandomResizedCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter()], p=color_jitter_p),
        transforms.RandomGrayscale(p=gray_scale_p),
        transforms.ToTensor(),
        transforms.Normalize(**normalize)
    ])
    return transform


def get_SimCLR_loader():
    dataset = SimCLRDataset(train_set, transform=get_SimCLR_transform(**SimCLR_kwargs['transform']))
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size * n_gpus,
        shuffle=True,
        num_workers=num_workers * n_gpus,
        pin_memory=True
    )
    return dataloader


def get_SL_loader():
    dataloader = DataLoader(
        train_set,
        batch_size=batch_size * n_gpus,
        shuffle=True,
        num_workers=num_workers * n_gpus,
        pin_memory=True
    )
    return dataloader


def get_test_loader():
    dataloader = DataLoader(
        test_set,
        batch_size=batch_size * n_gpus * 4,
        shuffle=False,
        num_workers=num_workers * n_gpus,
        pin_memory=True
    )
    return dataloader
