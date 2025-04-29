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
from torch.utils.data import DataLoader
import time
from tqdm import tqdm
import argparse
import nvtx

def train_rn50(
    data_dir: str,
    batch_size: int,
    num_epochs: int = 10,
    learning_rate: float = 0.001,
    num_classes: int = 10,
    device: str = "cuda"
) -> nn.Module:
    """
    Train a ResNet50 model on the specified dataset.
    
    Args:
        data_dir (str): Path to the dataset directory
        num_epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        learning_rate (float): Learning rate for the optimizer
        num_classes (int): Number of output classes
        device (str): Device to train on ('cuda' or 'cpu')
    
    Returns:
        nn.Module: Trained ResNet50 model
    """
    # Data transformations
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load dataset
    train_dataset = datasets.ImageFolder(
        root=data_dir,
        transform=transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    # Initialize model
    model = models.resnet50()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        epoch_start_time = time.time()
        total_images = 0
        
        train_iter = iter(train_loader)

        progress_bar = tqdm(range(len(train_loader)), desc=f'Epoch {epoch+1}/{num_epochs}')
        
        while True:
            try:
                nvtx.push_range("Get next batch")
                inputs, labels = next(train_iter)
                nvtx.pop_range()
            except StopIteration:
                nvtx.pop_range()
                break
            batch_start_time = time.time()
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            nvtx.push_range("Forward pass")
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            nvtx.pop_range()
            
            # Backward pass and optimize
            nvtx.push_range("Backward pass")
            loss.backward()
            optimizer.step()
            nvtx.pop_range()

            # Calculate throughput
            assert labels.size(0) == batch_size
            batch_end_time = time.time()
            batch_time = batch_end_time - batch_start_time
            batch_throughput = batch_size / batch_time
            
            # Update progress bar
            progress_bar.set_postfix({
                'imgs/s': f'{batch_throughput:.1f}'
            })
            
            total_images += batch_size
        
        # Calculate epoch statistics
        epoch_time = time.time() - epoch_start_time
        epoch_throughput = total_images / epoch_time
        
        print(f'\nEpoch {epoch+1} Summary:')
        print(f'Throughput: {epoch_throughput:.1f} images/second')
        print(f'Total Time: {epoch_time:.2f} seconds')
    
    return model

def infer_random_sample(model, data_dir, device):
    """
    Run inference on a random sample from the dataset.
    
    Args:
        model (nn.Module): Trained model
        data_dir (str): Path to dataset directory
        device (str): Device to run inference on
    """
    # Setup transforms
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load dataset
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    random_idx = torch.randint(0, len(dataset), (1,)).item()
    sample_image, ground_truth = dataset[random_idx]
    
    # Run inference
    model.eval()
    with torch.no_grad():
        input_batch = sample_image.unsqueeze(0).to(device)
        output = model(input_batch)
        _, predicted = torch.max(output, 1)
        
        # Get class names
        class_names = dataset.classes
        
        print("\nInference on random sample:")
        print(f"Ground truth: {class_names[ground_truth]}")
        print(f"Model prediction: {class_names[predicted.item()]}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train ResNet50 model')
    parser.add_argument('--data-dir', type=str, required=True,
                      help='Path to the dataset directory')
    parser.add_argument('--batch-size', type=int, default=64,
                      help='Batch size for training')
    
    args = parser.parse_args()
    
    model = train_rn50(
        data_dir=args.data_dir,
        batch_size=args.batch_size
    )

    # Run inference on a random sample
    infer_random_sample(model, args.data_dir, 'cuda:0')

    # Save the trained model
    torch.save(model.state_dict(), 'resnet50_trained.pth')
    print("Model saved to resnet50_trained.pth")
