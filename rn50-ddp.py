# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader, DistributedSampler
import time
from tqdm import tqdm
import argparse
import os
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

def setup(rank, world_size):
    """Initialize the distributed environment."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    """Clean up the distributed environment."""
    dist.destroy_process_group()

def train_rn50(
    rank: int,
    world_size: int,
    data_dir: str,
    num_epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    num_classes: int = 10,
    device: str = "cuda"
) -> nn.Module:
    """
    Train a ResNet50 model on the specified dataset using DDP.
    
    Args:
        rank (int): Process rank
        world_size (int): Total number of processes
        data_dir (str): Path to the dataset directory
        num_epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        learning_rate (float): Learning rate for the optimizer
        num_classes (int): Number of output classes
        device (str): Device to train on ('cuda' or 'cpu')
    
    Returns:
        nn.Module: Trained ResNet50 model
    """
    # Setup distributed environment
    setup(rank, world_size)
    
    # Data transformations
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load dataset with distributed sampler
    train_dataset = datasets.ImageFolder(
        root=data_dir,
        transform=transform
    )
    
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True
    )
    
    # Initialize model
    model = models.resnet50()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(rank)
    model = DDP(model, device_ids=[rank])
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        train_sampler.set_epoch(epoch)  # Set epoch for sampler
        epoch_start_time = time.time()
        total_images = 0
        
        if rank == 0:  # Only show progress bar on rank 0
            progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        else:
            progress_bar = train_loader
        
        for batch_idx, (inputs, labels) in enumerate(progress_bar):
            batch_start_time = time.time()
            inputs, labels = inputs.to(rank), labels.to(rank)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Calculate throughput
            batch_end_time = time.time()
            batch_time = batch_end_time - batch_start_time
            batch_throughput = labels.size(0) / batch_time
            
            if rank == 0:  # Only update progress bar on rank 0
                progress_bar.set_postfix({
                    'imgs/s': f'{batch_throughput:.1f}'
                })
            
            total_images += labels.size(0)
        
        if rank == 0:  # Only print summary on rank 0
            epoch_time = time.time() - epoch_start_time
            epoch_throughput = total_images / epoch_time
            
            print(f'\nEpoch {epoch+1} Summary:')
            print(f'Throughput: {epoch_throughput:.1f} images/second')
            print(f'Total Time: {epoch_time:.2f} seconds')
    
    cleanup()
    return model.module  # Return the actual model, not the DDP wrapper

def main():
    parser = argparse.ArgumentParser(description='Train ResNet50 model using DDP')
    parser.add_argument('--data-dir', type=str, required=True,
                      help='Path to the dataset directory')
    parser.add_argument('--num-epochs', type=int, default=10,
                      help='Number of training epochs (default: 10)')
    parser.add_argument('--batch-size', type=int, default=32,
                      help='Batch size for training (default: 32)')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                      help='Learning rate for the optimizer (default: 0.001)')
    parser.add_argument('--num-classes', type=int, default=10,
                      help='Number of output classes (default: 10)')
    parser.add_argument('--world-size', type=int, default=torch.cuda.device_count(),
                      help='Number of GPUs to use (default: all available)')
    
    args = parser.parse_args()
    
    # Launch distributed training
    mp.spawn(
        train_rn50,
        args=(args.world_size, args.data_dir, args.num_epochs, args.batch_size,
              args.learning_rate, args.num_classes),
        nprocs=args.world_size,
        join=True
    )

if __name__ == "__main__":
    main() 