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
import torch.distributed as dist
import torch.multiprocessing as mp
import time
from tqdm import tqdm
import argparse

def train_rn50(
    rank,
    world_size,
    data_dir: str,
    batch_size: int,
    num_epochs: int = 10,
    learning_rate: float = 0.001,
    num_classes: int = 10,
) -> nn.Module:
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    device = f"cuda:{rank}"

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_dataset = datasets.ImageFolder(
        root=data_dir,
        transform=transform
    )
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler
    )

    model = models.resnet50()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        epoch_start_time = time.time()
        total_images = 0

        train_sampler.set_epoch(epoch)
        if rank == 0:
            progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        else:
            progress_bar = train_loader

        for inputs, labels in progress_bar:
            batch_start_time = time.time()
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            batch_end_time = time.time()
            batch_time = batch_end_time - batch_start_time
            batch_throughput = labels.size(0) / batch_time

            if rank == 0:
                progress_bar.set_postfix({
                    'imgs/s': f'{batch_throughput:.1f}'
                })
            total_images += labels.size(0)

        epoch_time = time.time() - epoch_start_time
        epoch_throughput = total_images / epoch_time

        if rank == 0:
            print(f'\nEpoch {epoch+1} Summary:')
            print(f'Throughput: {epoch_throughput:.1f} images/second')
            print(f'Total Time: {epoch_time:.2f} seconds')

    if rank == 0:
        torch.save(model.module.state_dict(), 'resnet50_trained.pth')
        print("Model saved to resnet50_trained.pth")

    dist.destroy_process_group()
    return model.module if hasattr(model, "module") else model

def infer_random_sample(model, data_dir, device):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    random_idx = torch.randint(0, len(dataset), (1,)).item()
    sample_image, ground_truth = dataset[random_idx]
    model.eval()
    with torch.no_grad():
        input_batch = sample_image.unsqueeze(0).to(device)
        output = model(input_batch)
        _, predicted = torch.max(output, 1)
        class_names = dataset.classes
        print("\nInference on random sample:")
        print(f"Ground truth: {class_names[ground_truth]}")
        print(f"Model prediction: {class_names[predicted.item()]}")

def main_worker(rank, world_size, args):
    model = train_rn50(
        rank=rank,
        world_size=world_size,
        data_dir=args.data_dir,
        batch_size=args.batch_size
    )
    if rank == 0:
        infer_random_sample(model, args.data_dir, f'cuda:{rank}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train ResNet50 model')
    parser.add_argument('--data-dir', type=str, required=True,
                      help='Path to the dataset directory')
    parser.add_argument('--batch-size', type=int, default=64,
                      help='Batch size for training')
    parser.add_argument('--n-gpus', type=int, default=torch.cuda.device_count(),
                      help='Number of GPUs to use')
    args = parser.parse_args()

    world_size = args.n_gpus
    mp.spawn(main_worker, args=(world_size, args), nprocs=world_size)
