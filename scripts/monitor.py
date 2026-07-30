# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 15:49:08 2026

@author: salim
"""

import time
import psutil
import torch


class PerformanceMonitor:

    def __init__(self):

        self.training_start_time = None
        self.epoch_start_time = None
        self.batch_start_time = None

    ########################
    # Training
    ########################

    def start_training(self):
        self.training_start_time = time.perf_counter()

    def end_training(self):
        return time.perf_counter() - self.training_start_time

    ########################
    # Epoch
    ########################

    def start_epoch(self):
        self.epoch_start_time = time.perf_counter()

    def end_epoch(self):
        return time.perf_counter() - self.epoch_start_time

    ########################
    # Batch
    ########################

    def start_batch(self):
        self.batch_start_time = time.perf_counter()

    def end_batch(self):
        return time.perf_counter() - self.batch_start_time
    
    def get_cpu_usage(self):
        """
        Return CPU utilization in percent.
        """
        return psutil.cpu_percent(interval=None)
    
    def get_ram_usage(self):
        """
        Return RAM usage of the current process in GB.
        """
    
        process = psutil.Process()
    
        memory = process.memory_info().rss
    
        return memory / (1024 ** 3)
    
    def get_gpu_memory(self):
        
        if not torch.cuda.is_available():
            return 0.0
    
        return torch.cuda.memory_allocated() / (1024 ** 2)
