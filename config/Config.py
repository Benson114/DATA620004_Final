import os

os.environ['CUDA_LAUNCH_BLOCKING'] = "0,1,2,3,4,5,6,7"

import random
import numpy as np
import torch

seed = 3407
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

n_gpus = torch.cuda.device_count()

batch_size = 32
num_workers = 4
valid_size = 0.1

num_epochs = 100
