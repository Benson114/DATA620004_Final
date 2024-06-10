import torch.nn as nn

from config.Config import *


class SimpleCNN(nn.Module):
    def __init__(self, simple_cnn_kwargs):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(**simple_cnn_kwargs["conv2d_1"]),
            nn.ReLU(),
            nn.Conv2d(**simple_cnn_kwargs["conv2d_2"]),
            nn.ReLU(),
            nn.MaxPool2d(**simple_cnn_kwargs["maxpool2d_1"]),
            nn.Conv2d(**simple_cnn_kwargs["conv2d_3"]),
            nn.ReLU(),
            nn.Conv2d(**simple_cnn_kwargs["conv2d_4"]),
            nn.ReLU(),
            nn.MaxPool2d(**simple_cnn_kwargs["maxpool2d_2"])
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(**simple_cnn_kwargs["linear_1"]),
            nn.ReLU(),
            nn.Linear(**simple_cnn_kwargs["linear_2"])
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


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, patch_size, emb_size):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size),
            nn.Flatten(start_dim=2)
        )

    def forward(self, x):
        x = self.projection(x)  # [B, C, H, W] -> [B, N, D]
        x = x.permute(0, 2, 1)  # [B, N, D] -> [B, D, N]
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, emb_size, num_heads, dim_ffn, dropout):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=emb_size,
            num_heads=num_heads,
            dropout=dropout
        )
        self.norm1 = nn.LayerNorm(emb_size)
        self.norm2 = nn.LayerNorm(emb_size)
        self.ffn = nn.Sequential(
            nn.Linear(emb_size, dim_ffn),
            nn.ReLU(),
            nn.Linear(dim_ffn, emb_size)
        )

    def forward(self, x):
        x = x + self.attention(x, x, x)[0]
        x = self.norm1(x)
        x = x + self.ffn(x)
        x = self.norm2(x)
        return x


class VisionTransformer(nn.Module):
    def __init__(self, img_size,
                 in_channels, patch_size, emb_size,
                 num_layers, num_heads, dim_ffn, dropout,
                 num_classes):
        super().__init__()
        num_patches = (img_size // patch_size) ** 2

        self.patch_embedding = PatchEmbedding(in_channels, patch_size, emb_size)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, emb_size))

        self.encoders = nn.Sequential(*[
            TransformerEncoder(emb_size, num_heads, dim_ffn, dropout)
            for _ in range(num_layers)
        ])
        self.classifier = nn.Linear(emb_size, num_classes)

    def forward(self, x):
        x = self.patch_embedding(x)
        x = x + self.pos_embedding
        x = self.encoders(x)
        x = x.mean(dim=1)
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
