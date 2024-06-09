import os

os.environ['CUDA_VISIBLE_DEVICES'] = '6'

import random
import numpy as np
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

seed = 3407
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

n_gpus = torch.cuda.device_count()

batch_size = 32
num_workers = 4
valid_size = 0.1

num_epochs = 100

SimpleCNN_kwargs = {
    "conv2d_1": {
        "in_channels": 3,
        "out_channels": 32,
        "kernel_size": 3,
        "stride": 1,
        "padding": 1
    },
    "conv2d_2": {
        "in_channels": 32,
        "out_channels": 32,
        "kernel_size": 3,
        "stride": 1,
        "padding": 1
    },
    "maxpool2d_1": {
        "kernel_size": 2,
        "stride": 2
    },
    "conv2d_3": {
        "in_channels": 32,
        "out_channels": 64,
        "kernel_size": 3,
        "stride": 1,
        "padding": 1
    },
    "conv2d_4": {
        "in_channels": 64,
        "out_channels": 64,
        "kernel_size": 3,
        "stride": 1,
        "padding": 1
    },
    "maxpool2d_2": {
        "kernel_size": 2,
        "stride": 2
    },
    "linear_1": {
        "in_features": 64 * 8 * 8,
        "out_features": 512
    },
    "linear_2": {
        "in_features": 512,
        "out_features": 100
    }
}

ViT_kwargs = {
    "img_size": 32,
    "in_channels": 3,
    "patch_size": 4,
    "emb_size": 256,
    "num_layers": 4,
    "num_heads": 8,
    "dim_ffn": 1024,
    "dropout": 0.1,
    "num_classes": 100
}
