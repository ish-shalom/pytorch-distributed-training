# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 00:50:39 2026

@author: salim
"""

import torch
import torch.distributed as dist

def train_one_epoch(
    model,
    dataloader,
    optimizer,
    criterion,
    device,
):

    model.train()

    running_loss = 0.0
    num_samples = 0

    for batch_idx, (images, labels) in enumerate(dataloader):

        images = images.to(device)
        labels = labels.to(device)

        predictions = model(images)

        loss = criterion(predictions, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        batch_size = images.size(0)

        running_loss += loss.item() * batch_size
        num_samples += batch_size

    loss_tensor = torch.tensor([running_loss, num_samples], device=device, dtype=torch.float64)
    if dist.is_initialized() and dist.get_world_size() > 1:
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
    global_loss = (loss_tensor[0] / loss_tensor[1])

    return global_loss.item()


def evaluate(
    model,
    dataloader,
    criterion,
    device,
):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in dataloader:

            images = images.to(device)
            labels = labels.to(device)

            predictions = model(images)

            loss = criterion(predictions, labels)

            running_loss += loss.item()

            predicted_classes = predictions.argmax(dim=1)

            correct += (predicted_classes == labels).sum().item()
            total += labels.size(0)

    average_loss = running_loss / len(dataloader)
    accuracy = correct / total

    return average_loss, accuracy


