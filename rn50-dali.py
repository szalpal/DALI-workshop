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
import nvidia.dali.fn as fn
from nvidia.dali.pipeline import pipeline_def
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import DALIGenericIterator
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
import time
from tqdm import tqdm
import argparse

@pipeline_def
def dali_pipeline(data_dir):
    images, labels = fn.readers.file(file_root=data_dir, random_shuffle=True)
    images = fn.decoders.image(images, device="mixed", output_type=types.RGB)
    labels = fn.cast(labels, dtype=types.INT64).gpu()
    images = fn.resize(images, resize_x=224, resize_y=224)
    images = fn.crop_mirror_normalize(
        images,
        dtype=types.FLOAT,
        output_layout="CHW",
        crop=(224, 224),
        mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
        std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
    )
    return images, labels

def dali(data_dir, batch_size):
    pipe = dali_pipeline(data_dir, batch_size=batch_size, num_threads=32, device_id=0)
    iterator = DALIGenericIterator(pipe, ["images", "labels"])
    return iterator

def train_rn50(
    data_dir: str,
    batch_size: int,
    num_epochs: int = 10,
    learning_rate: float = 0.001,
    num_classes: int = 10,
    device: str = "cuda",
) -> nn.Module:
    train_loader = dali(data_dir, batch_size)

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

        for data in train_loader:
            batch_start_time = time.time()
            inputs, labels = data[0]["images"], data[0]["labels"].flatten()

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Calculate throughput
            assert labels.size(0) == batch_size, f"Batch size mismatch: got {labels.size(0)}, expected {batch_size}"
            batch_end_time = time.time()
            batch_time = batch_end_time - batch_start_time
            batch_throughput = batch_size / batch_time

            print(f"Throughput: {batch_throughput:.1f} images/second")

            total_images += batch_size

        # Calculate epoch statistics
        epoch_time = time.time() - epoch_start_time
        epoch_throughput = total_images / epoch_time

        print(f"\nEpoch {epoch+1} Summary:")
        print(f"Throughput: {epoch_throughput:.1f} images/second")
        print(f"Total Time: {epoch_time:.2f} seconds")

    return model

def infer_random_sample(model, data_dir, device):
    # Setup transforms
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

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
    parser = argparse.ArgumentParser(description="Train ResNet50 model")
    parser.add_argument(
        "--data-dir", type=str, required=True, help="Path to the dataset directory"
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Batch size for training"
    )

    args = parser.parse_args()

    model = train_rn50(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
    )

    # Run inference on a random sample
    infer_random_sample(model, args.data_dir, "cuda:0")

    # Save the trained model
    torch.save(model.state_dict(), "resnet50_trained.pth")
    print("Model saved to resnet50_trained.pth")
