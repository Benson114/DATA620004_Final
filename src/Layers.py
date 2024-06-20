import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    def __init__(self, img_size, in_channels, patch_size, emb_size):
        super().__init__()
        self.projection = nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.pos_embedding = nn.Parameter(torch.randn(1, 1 + (img_size // patch_size) ** 2, emb_size))

    def forward(self, x):
        x = self.projection(x)
        x = x.flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding  # [B, 1 + N, emb_size]
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, emb_size, num_heads, dim_ffn, dropout):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=emb_size,
            num_heads=num_heads,
            dropout=dropout
        )
        self.LN1 = nn.LayerNorm(emb_size)
        self.LN2 = nn.LayerNorm(emb_size)
        self.ffn = nn.Sequential(
            nn.Linear(emb_size, dim_ffn),
            nn.ReLU(),
            nn.Linear(dim_ffn, emb_size),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.LN1(x)
        x = x + self.attention(x, x, x)[0]
        x = self.LN2(x)
        x = self.ffn(x)
        return x
