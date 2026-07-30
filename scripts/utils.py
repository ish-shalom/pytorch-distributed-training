# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 21:54:55 2026

@author: salim
"""

import matplotlib.pyplot as plt
import torch
import random
import numpy as np

def set_seed(seed: int = 42):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

def count_parameters(model):
    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


def denormalize_image(image, label):
    # Image denormalization and visualization
    
    mean = 0.1307
    std = 0.3081

    image = image * std + mean

    plt.imshow(image.squeeze(), cmap="gray")
    plt.title(f"Label: {label}")
    plt.show()
