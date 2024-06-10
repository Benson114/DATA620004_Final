import torch.nn.functional as F
from torch.nn import CrossEntropyLoss


def CutmixCriterion(outputs, labels):
    targets1, targets2, lam = labels[0], labels[1], labels[2]
    criterion = CrossEntropyLoss()
    loss1 = criterion(outputs, targets1)
    loss2 = criterion(outputs, targets2)
    return lam * loss1 + (1 - lam) * loss2
