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

num_classes = 100

num_workers = 4
num_epochs_full = 120
num_epochs_pretrain = 100
batch_size = 32
lr = 1e-3

SimCLR_kwargs = {
    "pj_head_dim": [512, 256, 128],
    "transform": {
        "color_jitter_p": 0.8,
        "gray_scale_p": 0.2,
        "normalize": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225]
        }
    }
}
