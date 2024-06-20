import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    def __init__(self, img_size, in_channels, patch_size, emb_size, dropout):
        super().__init__()
        self.patch_embedding = nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.pos_embedding = nn.Parameter(torch.randn(1, 1 + (img_size // patch_size) ** 2, emb_size))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)

        x = self.patch_embedding(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        x = torch.cat((cls_tokens, x), dim=1)
        emb = x + self.pos_embedding  # [B, 1 + N, emb_size]
        emb = self.dropout(emb)
        return emb


class Attention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads

        self.q = nn.Linear(emb_size, emb_size)
        self.k = nn.Linear(emb_size, emb_size)
        self.v = nn.Linear(emb_size, emb_size)

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        b, n, d = x.size()
        assert d == self.emb_size

        q_layer = self.q(x)
        k_layer = self.k(x)
        v_layer = self.v(x)

        q_layer = self._reshape_heads(q_layer)
        k_layer = self._reshape_heads(k_layer)
        v_layer = self._reshape_heads(v_layer)

        attn_scores = torch.matmul(q_layer, k_layer.transpose(-2, -1))
        attn_scores = self.softmax(attn_scores)
        attn_scores = self.dropout(attn_scores)

        out = torch.matmul(attn_scores, v_layer)
        out = self._reshape_heads_back(out)

        return out, attn_scores

    def _reshape_heads(self, x):
        b, n, d = x.size()
        reduced_dim = d // self.num_heads
        assert reduced_dim * self.num_heads == d

        out = x.reshape(b, n, self.num_heads, reduced_dim)
        out = out.permute(0, 2, 1, 3)
        out = out.reshape(-1, n, reduced_dim)
        return out

    def _reshape_heads_back(self, x):
        b, n, d = x.size()
        b = b // self.num_heads

        out = x.reshape(b, self.num_heads, n, d)
        out = out.permute(0, 2, 1, 3)
        out = out.reshape(b, n, self.emb_size)
        return out


class FFN(nn.Module):
    def __init__(self, emb_size, dim_ffn, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.fc1 = nn.Linear(emb_size, dim_ffn)
        self.fc2 = nn.Linear(dim_ffn, emb_size)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        out = self.dropout(self.activation(self.fc1(x)))
        out = self.fc2(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, emb_size, num_heads, dim_ffn, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.ln1 = nn.LayerNorm(emb_size, eps=1e-6)
        self.ln2 = nn.LayerNorm(emb_size, eps=1e-6)
        self.attention = Attention(emb_size, num_heads, dropout)
        self.ffn = FFN(emb_size, dim_ffn, dropout)

    def forward(self, x):
        res = x
        out = self.ln1(x)
        out, _ = self.attention(out)
        out = out + res

        res = out
        out = self.ln2(out)
        out = self.ffn(out)
        out = out + res
        return out


class TransformerEncoder(nn.Module):
    def __init__(self, emb_size, num_layers, num_heads, dim_ffn, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.layers = nn.ModuleList(
            [TransformerBlock(emb_size, num_heads, dim_ffn, dropout) for _ in range(num_layers)]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class ClassificationHead(nn.Module):
    def __init__(self, emb_size, num_classes, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_classes = num_classes
        self.fc1 = nn.Linear(emb_size, emb_size // 2)
        self.fc2 = nn.Linear(emb_size // 2, num_classes)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = self.dropout(self.activation(self.fc1(x)))
        out = self.fc2(out)
        return out
