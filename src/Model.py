import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


class ResNet18(nn.Module):
    def __init__(self, num_classes, pretrain_type):
        super(ResNet18, self).__init__()

        self.num_classes = num_classes
        self.pretrain_type = pretrain_type

        self.feature_net = None
        self.linear_classifier = nn.Linear(512, self.num_classes)

        if pretrain_type == "None" or pretrain_type == "SSL":
            self.feature_net = resnet18()
            self.feature_net.fc = nn.Identity()
        elif pretrain_type == "SL":
            self.feature_net = resnet18(weights='IMAGENET1K_V1')
            self.feature_net.fc = nn.Identity()
            for param in self.feature_net.parameters():
                param.requires_grad = False
        else:
            raise ValueError("pretrain_type must be 'SSL' or 'SL' or 'None'")

    def forward(self, x):
        feats = self.feature_net(x)
        feats = torch.flatten(feats, 1)
        out = self.linear_classifier(feats)
        return out

    def save(self, parent_dir):
        os.makedirs(parent_dir, exist_ok=True)
        torch.save(self.state_dict(), parent_dir + f"/{self.pretrain_type}.pth")
        return self

    def load(self, parent_dir):
        self.load_state_dict(torch.load(parent_dir + f"/{self.pretrain_type}.pth"))
        self.eval()
        return self


class SimCLR(nn.Module):
    def __init__(self, base_model, pj_head_dim):
        super(SimCLR, self).__init__()

        self.base_model = base_model
        self.projection_head = nn.Sequential(
            nn.Linear(pj_head_dim[0], pj_head_dim[1]),
            nn.ReLU(),
            nn.Linear(pj_head_dim[1], pj_head_dim[2])
        )

    def forward(self, x):
        feats = self.base_model(x)
        feats = torch.flatten(feats, 1)
        out = self.projection_head(feats)
        return out


def SimCLR_Loss(z1, z2, temperature=0.5):
    """
    z1 和 z2 分别是对同一批量的图像应用两次数据增强后的特征表示
    temperature 是温度参数
    """
    z = torch.cat([z1, z2], dim=0)  # [2*B, D]
    sim = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)  # [2*B, 2*B]
    mask = ~torch.eye(z.size(0), dtype=bool, device=z.device)
    masked_sim = sim[mask].view(z.size(0), -1) / temperature  # [2*B, 2*B-1]

    labels = torch.arange(z.size(0)).to(z.device)  # [2*B]
    labels[z1.size(0):] -= z1.size(0)
    labels[:z1.size(0)] += z1.size(0) - 1

    loss = F.cross_entropy(masked_sim, labels)
    return loss
