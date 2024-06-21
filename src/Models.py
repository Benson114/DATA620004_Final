from config.Config import *
from src.Layers import *


class SimpleCNN(nn.Module):
    def __init__(self, kwargs):
        super(SimpleCNN, self).__init__()
        self.name = "SimpleCNN"

        self.features = nn.Sequential(
            nn.Conv2d(**kwargs["conv2d_1"]), nn.BatchNorm2d(kwargs["bn_1"]),
            nn.ReLU(), nn.MaxPool2d(**kwargs["maxpool2d"]),
            nn.Conv2d(**kwargs["conv2d_2"]), nn.BatchNorm2d(kwargs["bn_2"]),
            nn.ReLU(), nn.MaxPool2d(**kwargs["maxpool2d"]),
            nn.Conv2d(**kwargs["conv2d_3"]), nn.BatchNorm2d(kwargs["bn_3"]),
            nn.ReLU(), nn.MaxPool2d(**kwargs["maxpool2d"]),
            nn.Conv2d(**kwargs["conv2d_4"]), nn.BatchNorm2d(kwargs["bn_4"]),
            nn.ReLU(), nn.MaxPool2d(**kwargs["maxpool2d"])
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(**kwargs["linear_1"]), nn.ReLU(), nn.Dropout(kwargs["dropout"]),
            nn.Linear(**kwargs["linear_2"]), nn.ReLU(), nn.Dropout(kwargs["dropout"]),
            nn.Linear(**kwargs["linear_3"])
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    def save(self, parent_dir, ckpt_name):
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)
        path = os.path.join(parent_dir, ckpt_name)
        torch.save(self.state_dict(), str(path))

    def load(self, parent_dir, ckpt_name):
        path = os.path.join(parent_dir, ckpt_name)
        self.load_state_dict(torch.load(str(path)))
        self.eval()


class VisionTransformer(nn.Module):
    def __init__(self, img_size,
                 in_channels, patch_size, emb_size,
                 num_layers, num_heads, dim_ffn, dropout,
                 num_classes):
        super().__init__()
        self.name = "VisionTransformer"

        self.patch_embedding = PatchEmbedding(img_size, in_channels, patch_size, emb_size, dropout)
        self.transformer = TransformerEncoder(emb_size, num_layers, num_heads, dim_ffn, dropout)
        self.classifier = ClassificationHead(emb_size, num_classes, dropout)

    def forward(self, x):
        embedding = self.patch_embedding(x)
        out = self.transformer(embedding)
        cls = out[:, 0]
        logits = self.classifier(cls)
        return logits

    def save(self, parent_dir, ckpt_name):
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)
        path = os.path.join(parent_dir, ckpt_name)
        torch.save(self.state_dict(), str(path))

    def load(self, parent_dir, ckpt_name):
        path = os.path.join(parent_dir, ckpt_name)
        self.load_state_dict(torch.load(str(path)))
        self.eval()
