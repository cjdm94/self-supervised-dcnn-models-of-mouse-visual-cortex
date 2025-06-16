import torch
import random
import numpy as np

GLOBAL_SEED = 42


def set_global_seed(seed: int = GLOBAL_SEED):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
