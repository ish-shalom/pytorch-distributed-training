# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 17:25:24 2026

@author: salim
"""

import torch.distributed as dist


def setup(backend):
    """
    Initialize distributed training environment.
    Values are provided by torchrun through environment variables.
    """
        
    dist.init_process_group(
        backend=backend,
        init_method="env://",
    )


def cleanup():

    if dist.is_initialized():
        dist.destroy_process_group()
