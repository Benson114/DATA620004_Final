import torchvision
from torchvision import transforms


def CIFAR100_Dataset():
    def data_augmentation():
        return [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(90),
            transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)
        ]

    transform_train = transforms.Compose([
        *data_augmentation(),  # 训练数据需要数据增强
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    train_set = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    test_set = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)

    return train_set, test_set
