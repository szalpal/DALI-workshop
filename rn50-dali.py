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

print("[DEBUG] Imports completed")

@pipeline_def
def dali_pipeline(data_dir):
    print(f"[DEBUG] Creating DALI pipeline for data_dir: {data_dir}")
    images, labels = fn.readers.file(file_root=data_dir, random_shuffle=True)
    print("[DEBUG] DALI file reader created")
    images = fn.decoders.image(images, device="mixed", output_type=types.RGB)
    print("[DEBUG] DALI image decoder created")
    labels = fn.cast(labels, dtype=types.INT64).gpu()
    print("[DEBUG] DALI labels cast to INT64 and moved to GPU")
    images = fn.resize(images, resize_x=224, resize_y=224)
    print("[DEBUG] DALI image resize to 224x224")
    images = fn.crop_mirror_normalize(
        images,
        dtype=types.FLOAT,
        output_layout="CHW",
        crop=(224, 224),
        mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
        std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
    )
    print("[DEBUG] DALI crop, mirror, normalize applied")
    return images, labels


def dali(data_dir, batch_size):
    print(f"[DEBUG] Initializing DALI pipeline with batch_size: {batch_size}")
    pipe = dali_pipeline(data_dir, batch_size=batch_size, num_threads=1, device_id=0)
    print("[DEBUG] DALI pipeline instance created")
    iterator = DALIGenericIterator(pipe, ["images", "labels"])
    print("[DEBUG] DALIGenericIterator created")
    return iterator


def train_rn50(
    data_dir: str,
    batch_size: int,
    num_epochs: int = 10,
    learning_rate: float = 0.001,
    num_classes: int = 10,
    device: str = "cuda",
) -> nn.Module:
    print(f"[DEBUG] Starting train_rn50 with data_dir={data_dir}, batch_size={batch_size}, num_epochs={num_epochs}, learning_rate={learning_rate}, num_classes={num_classes}, device={device}")
    train_loader = dali(data_dir, batch_size)
    print("[DEBUG] DALI train_loader created")

    # Initialize model
    print("[DEBUG] Initializing ResNet50 model")
    model = models.resnet50()
    print("[DEBUG] ResNet50 model created")
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    print(f"[DEBUG] Replaced final FC layer with {num_classes} outputs")
    model = model.to(device)
    print(f"[DEBUG] Model moved to device: {device}")

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    print("[DEBUG] CrossEntropyLoss created")
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    print("[DEBUG] Adam optimizer created")

    # Training loop
    for epoch in range(num_epochs):
        print(f"[DEBUG] Starting epoch {epoch+1}/{num_epochs}")
        model.train()
        epoch_start_time = time.time()
        total_images = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for batch_idx, data in enumerate(progress_bar):
            print(f"\n[DEBUG] Epoch {epoch+1}, Batch {batch_idx+1} - Starting batch")
            batch_start_time = time.time()
            print(f"[DEBUG] Raw batch data keys: {list(data[0].keys())}")
            inputs, labels = data[0]["images"], data[0]["labels"].flatten()
            print(f"[DEBUG] Inputs shape: {inputs.shape}, dtype: {inputs.dtype}, device: {inputs.device}")
            print(f"[DEBUG] Labels shape: {labels.shape}, dtype: {labels.dtype}, device: {labels.device}")

            # Zero the parameter gradients
            optimizer.zero_grad()
            print("[DEBUG] Gradients zeroed")

            # Forward pass
            outputs = model(inputs)
            print(f"[DEBUG] Forward pass done. Outputs shape: {outputs.shape}, dtype: {outputs.dtype}, device: {outputs.device}")
            loss = criterion(outputs, labels)
            print(f"[DEBUG] Loss computed: {loss.item()}")

            # Backward pass and optimize
            loss.backward()
            print("[DEBUG] Backward pass done")
            optimizer.step()
            print("[DEBUG] Optimizer step done")

            # Calculate throughput
            assert labels.size(0) == batch_size, f"[DEBUG] Batch size mismatch: got {labels.size(0)}, expected {batch_size}"
            batch_end_time = time.time()
            batch_time = batch_end_time - batch_start_time
            batch_throughput = batch_size / batch_time

            print(f"[DEBUG] Batch time: {batch_time:.4f} seconds")
            print(f"[DEBUG] Throughput: {batch_throughput:.1f} images/second")

            # Update progress bar
            progress_bar.set_postfix({"imgs/s": f"{batch_throughput:.1f}", "loss": f"{loss.item():.4f}"})

            total_images += batch_size

        # Calculate epoch statistics
        epoch_time = time.time() - epoch_start_time
        epoch_throughput = total_images / epoch_time

        print(f"\n[DEBUG] Epoch {epoch+1} Summary:")
        print(f"[DEBUG] Throughput: {epoch_throughput:.1f} images/second")
        print(f"[DEBUG] Total Time: {epoch_time:.2f} seconds")

    print("[DEBUG] Training complete")
    return model


def infer_random_sample(model, data_dir, device):
    print(f"[DEBUG] Running inference on a random sample from {data_dir} using device {device}")
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
    print("[DEBUG] Loading ImageFolder dataset for inference")
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    print(f"[DEBUG] Dataset loaded with {len(dataset)} samples, classes: {dataset.classes}")
    random_idx = torch.randint(0, len(dataset), (1,)).item()
    print(f"[DEBUG] Randomly selected index: {random_idx}")
    sample_image, ground_truth = dataset[random_idx]
    print(f"[DEBUG] Sample image shape: {sample_image.shape}, ground truth label: {ground_truth}")

    # Run inference
    model.eval()
    with torch.no_grad():
        input_batch = sample_image.unsqueeze(0).to(device)
        print(f"[DEBUG] Input batch shape: {input_batch.shape}, device: {input_batch.device}")
        output = model(input_batch)
        print(f"[DEBUG] Model output: {output}")
        _, predicted = torch.max(output, 1)
        print(f"[DEBUG] Predicted class index: {predicted.item()}")

        # Get class names
        class_names = dataset.classes

        print("\nInference on random sample:")
        print(f"Ground truth: {class_names[ground_truth]}")
        print(f"Model prediction: {class_names[predicted.item()]}")


if __name__ == "__main__":
    print("[DEBUG] Parsing command line arguments")
    parser = argparse.ArgumentParser(description="Train ResNet50 model")
    parser.add_argument(
        "--data-dir", type=str, required=True, help="Path to the dataset directory"
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Batch size for training"
    )

    args = parser.parse_args()
    print(f"[DEBUG] Arguments received: data_dir={args.data_dir}, batch_size={args.batch_size}")

    model = train_rn50(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
    )

    # Run inference on a random sample
    infer_random_sample(model, args.data_dir, "cuda:0")

    # Save the trained model
    torch.save(model.state_dict(), "resnet50_trained.pth")
    print("[DEBUG] Model saved to resnet50_trained.pth")
