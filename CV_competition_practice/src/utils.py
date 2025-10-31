# src/utils.py

import torch
import numpy as np
import random
import os


def set_seed(seed=42):
    """
    Set random seed for reproducibility (MPS optimized)

    Args:
        seed (int): Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # MPS seed for Apple Silicon
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

    # Environment variable
    os.environ['PYTHONHASHSEED'] = str(seed)

    print(f"Seed set: {seed} (MPS)")
