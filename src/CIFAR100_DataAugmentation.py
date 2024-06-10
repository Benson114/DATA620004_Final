from torch.utils.data.dataloader import default_collate

from config.Config import *


def cutmix_batch(batch, alpha, prob):
    images, labels = batch

    if np.random.rand() > prob:
        labels = [labels, labels, 1.0]
        return images, labels

    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0

    indices = torch.randperm(images.size(0))

    img_h, img_w = images.size(2), images.size(3)
    cx, cy = np.random.randint(img_w), np.random.randint(img_h)

    cut_ratio = np.sqrt(1. - lam)
    cut_w, cut_h = int(img_w * cut_ratio), int(img_h * cut_ratio)

    bbx1 = np.clip(cx - cut_w // 2, 0, img_w)
    bby1 = np.clip(cy - cut_h // 2, 0, img_h)
    bbx2 = np.clip(cx + cut_w // 2, 0, img_w)
    bby2 = np.clip(cy + cut_h // 2, 0, img_h)

    images[:, :, bby1:bby2, bbx1:bbx2] = images[indices, :, bby1:bby2, bbx1:bbx2]
    lam = 1 - (bbx2 - bbx1) * (bby2 - bby1) / float(img_w * img_h)

    labels = [labels, labels[indices], lam]
    return images, labels


def collate_fn_cutmix(batch):
    batch = default_collate(batch)
    batch = cutmix_batch(batch, **cutmix_kwargs)
    return batch
