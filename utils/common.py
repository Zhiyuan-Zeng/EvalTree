
import os
import torch
import random
import numpy as np


def manual_seed(args_or_seed : int, fix_cudnn = False) :
    random.seed(args_or_seed)
    np.random.seed(args_or_seed)
    torch.manual_seed(args_or_seed)
    torch.cuda.manual_seed_all(args_or_seed)
    os.environ["PYTHONHASHSEED"] = str(args_or_seed)
    if fix_cudnn :
        torch.backends.cudnn.deterministic = True  # noqa
        torch.backends.cudnn.benchmark = False  # noqa