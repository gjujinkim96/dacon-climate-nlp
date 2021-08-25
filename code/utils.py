import torch
import random
import numpy as np

from pathlib import Path

def set_random(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def get_unique_name(dir_path, name, extension='csv'):
    dir = Path(dir_path)
    cur_name = dir / f'{name}.{extension}'
    if cur_name.exists():
        n = 1
        cur_name = dir / f'{name}_{n}.{extension}'
        while cur_name.exists():
            n += 1
            cur_name = dir / f'{name}_{n}.{extension}'
    
    return str(cur_name)
        
