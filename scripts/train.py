# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 21:58:10 2026

@author: salim
"""

import torch
from torch import nn
from torch.optim import Adam
import os
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import json
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import sys

from model import CNN
from utils import (
    count_parameters,
    set_seed,
    #denormalize_image,
)
from dataset import (
    get_dataloaders,
)
from engine import (
    train_one_epoch,
    evaluate,
)
from monitor import PerformanceMonitor
from distributed import setup, cleanup

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def main(config_path):
    with open(config_path) as f:
        config = json.load(f)
    
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    config["distributed"]["world_size"] = world_size
    
    if torch.cuda.is_available():
        backend = "nccl"
    else:
        backend = "gloo"
    config["distributed"]["backend"] = backend

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")
    config["distributed"]["device"] = device.type

    print(f"Rank {rank}/{world_size} | Local rank {local_rank} | Backend {backend}")
    
    set_seed(seed=config["training"]["seed"])
    if world_size > 1:
        setup(backend)
        dist.barrier()
    #torch.autograd.set_detect_anomaly(True)
    
    monitor = PerformanceMonitor()

    batch_size = config["training"]["batch_size"]
    learning_rate = config["training"]["learning_rate"]
    num_epochs = config["training"]["num_epochs"]
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    experiment_name = (
        f"CNN_MNIST_dv_{device.type}_ps_{world_size}_bs_{batch_size}_lr_{learning_rate}"
    )
    run_name = (f"{experiment_name}_{timestamp}")
    
    if rank == 0:
        print(f"Using device: {device}")
        
        os.makedirs(os.path.join(config["logging"]["checkpoints_dir"], experiment_name), exist_ok=True)
        checkpoint_path = os.path.join(config["logging"]["checkpoints_dir"], 
                                       experiment_name, 
                                       "checkpoint.pth"
        )
    
        log_dir = os.path.join(config["logging"]["runs_dir"], run_name)
        os.makedirs(log_dir, exist_ok=True)
    
        # Expanded config file for logging
        with open(os.path.join(log_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=4)
    
        if config["logging"]["tensorboard"]:
            writer = SummaryWriter(log_dir=log_dir)
    
    train_loader, test_loader, train_sampler = get_dataloaders(
        batch_size=batch_size,
        num_workers=config["distributed"]["num_workers_per_process"],
        root=config["dataset"]["root"],
        distributed=(world_size > 1),
        rank=rank,
        world_size=world_size,
    )

    model = CNN(
        in_channels=config["model"]["in_channels"],
        num_classes=config["model"]["num_classes"]
    ).to(device)
    
    if rank == 0:
        print(model)
        print(f"Trainable parameters: {count_parameters(model)}")
        monitor.start_training()

    criterion = nn.CrossEntropyLoss()

    start_epoch = 0
    best_accuracy = 0.0
        
    # Each process receives a copy of the model
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    
    # Model loading if a previous checkpoint is available
    # if rank == 0 and os.path.exists(checkpoint_path):
    
    #     checkpoint = torch.load(
    #         checkpoint_path,
    #         map_location=device
    #     )
    
    #     model.load_state_dict(checkpoint["model_state_dict"])
    #     optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    #     start_epoch = checkpoint["epoch"] + 1
    #     best_accuracy = checkpoint["best_accuracy"]
    
    #     print(f"Resuming from epoch {start_epoch}.")

    for epoch in range(start_epoch, num_epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        if rank == 0:
            monitor.start_epoch()

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
        )

        if rank == 0:
            val_loss, val_accuracy = evaluate(
                model.module if world_size > 1 else model,
                test_loader,
                criterion,
                device
            )
            
            epoch_time = monitor.end_epoch()
            print(
                f"Epoch {epoch + 1}/{num_epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_accuracy * 100:.2f}% | "
                f"Epoch time: {epoch_time:.1f} s"
            )
            
            if config["logging"]["tensorboard"]:
                writer.add_scalar("Loss/train", train_loss, epoch)
                writer.add_scalar("Loss/validation", val_loss, epoch)
                writer.add_scalar("Accuracy/validation", val_accuracy, epoch)
                writer.add_scalar("Performance/Epoch time (s)", epoch_time, epoch)
                writer.add_scalar("Performance/CPU (%)", monitor.get_cpu_usage(), epoch)
                writer.add_scalar("Performance/RAM (GB)", monitor.get_ram_usage(), epoch)
                
                if torch.cuda.is_available():
                    writer.add_scalar("Performance/GPU memory (MB)", monitor.get_gpu_memory(), epoch)
                    
            # Checkpoint saving
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.module.state_dict() if world_size > 1 else model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_accuracy": best_accuracy,
                },
                checkpoint_path,
            )

            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
    
                torch.save(
                    model.module.state_dict() if world_size > 1 else model.state_dict(),
                    os.path.join(config["logging"]["checkpoints_dir"], experiment_name, "best_model.pth")
                )
                print("Best model saved.")
        else:
            val_loss = 0
            val_accuracy = 0
        #dist.barrier()
    
    if rank == 0:
        total_training_time = monitor.end_training()
        print(f"Total training time: {total_training_time:.2f} s")
        # Number of images processed per second
        throughput = len(train_loader.dataset) / total_training_time
        
        if config["logging"]["tensorboard"]:
            writer.add_scalar("Performance/train_time", total_training_time, num_epochs) 
            writer.add_scalar("Performance/throughput", throughput, num_epochs)
            writer.close()
    
    cleanup()


if __name__ == "__main__":
    config_path = "configs/config.json"
    try:
        main(config_path)
    finally:
        if dist.is_initialized():
            cleanup()
        else:
            pass
