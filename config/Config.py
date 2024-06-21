import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

import random
import numpy as np
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

seed = 3407
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

n_gpus = torch.cuda.device_count()

batch_size = 64
num_workers = 8
valid_rate = 0.1

cutmix_kwargs = {
    "alpha": 1.0,
    "prob": 0.5,
}

num_epochs = 200

optimizer_kwargs = {
    "lr": 1e-3,
    "weight_decay": 3e-4,
}

SimpleCNN_kwargs = {
    "conv2d_1": {
        "in_channels": 3,
        "out_channels": 64,
        "kernel_size": 3,
        "stride": 1,
        "padding": 1,
    },
    "conv2d_2": {
        "in_channels": 64,
        "out_channels": 128,
        "kernel_size": 3,
        "stride": 1,
        "padding": 1,
    },
    "conv2d_3": {
        "in_channels": 128,
        "out_channels": 256,
        "kernel_size": 3,
        "stride": 1,
        "padding": 1,
    },
    "conv2d_4": {
        "in_channels": 256,
        "out_channels": 512,
        "kernel_size": 3,
        "stride": 1,
        "padding": 1,
    },
    "maxpool2d": {
        "kernel_size": 2,
        "stride": 2,
    },
    "linear_1": {
        "in_features": 2048,
        "out_features": 4096,
    },
    "linear_2": {
        "in_features": 4096,
        "out_features": 4096,
    },
    "linear_3": {
        "in_features": 4096,
        "out_features": 100,
    },
    "bn_1": 64,
    "bn_2": 128,
    "bn_3": 256,
    "bn_4": 512,
    "dropout": 0.5,
}

ViT_kwargs = {
    "img_size": 32,
    "in_channels": 3,
    "patch_size": 4,
    "emb_size": 512,
    "num_layers": 16,
    "num_heads": 16,
    "dim_ffn": 1024,
    "dropout": 0.1,
    "num_classes": 100,
}
