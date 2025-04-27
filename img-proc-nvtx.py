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

import nvidia.dali.fn as fn
from nvidia.dali.pipeline import pipeline_def
import nvidia.dali.types as types
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from nvidia.dali.plugin.pytorch import DALIGenericIterator
import time
from tqdm import tqdm
import argparse
import nvtx

@pipeline_def
def dali_pipeline(data_dir):
    images, labels = fn.readers.file(file_root=data_dir, random_shuffle=True)
    images = fn.decoders.image(images, device="mixed", output_type=types.RGB)
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
    pipe = dali_pipeline(data_dir, batch_size=256, num_threads=1, device_id=0)
    iterator = DALIGenericIterator(pipe, ["images", "labels"])
    return iterator


def pytorch(data_dir, batch_size):
    transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Load dataset
    train_dataset = datasets.ImageFolder(root=data_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    return train_loader


def test_iterator_performance(iterator, desc, batch_size, num_iterations=100):
    """
    Test the performance of a data iterator.

    Args:
        iterator: Data iterator to test
        num_iterations (int): Number of iterations to run
    """
    total_time = 0
    total_images = 0

    # Warm up iterations
    for _ in range(10):
        nvtx.push_range("Warm up")
        next(iter(iterator))
        nvtx.pop_range()

    # Timing loop
    progress_bar = tqdm(range(num_iterations), desc=desc)
    for _ in progress_bar:
        start_time = time.time()
        nvtx.push_range("Get next batch")
        batch = next(iterator)
        nvtx.pop_range()
        end_time = time.time()

        batch_time = end_time - start_time
        throughput = batch_size / batch_time

        total_time += batch_time
        total_images += batch_size

        progress_bar.set_postfix({"imgs/s": f"{throughput:.1f}"})

    avg_throughput = total_images / total_time
    print(f"Average throughput: {avg_throughput:.1f} images/second")
    print(f"Total images processed: {total_images}")
    print(f"Total time: {total_time:.2f} seconds\n")


def main(args):
    pytorch_pipeline = pytorch(args.data_dir, args.batch_size)
    test_iterator_performance(iter(pytorch_pipeline), "PyTorch iterator performance", args.batch_size)

    dali_pipeline = dali(args.data_dir, args.batch_size)
    test_iterator_performance(dali_pipeline, "DALI iterator performance", args.batch_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ResNet50 preprocessing pipelines")
    parser.add_argument(
        "--data-dir", type=str, required=True, help="Path to the dataset directory"
    )
    parser.add_argument(
        "--batch-size", type=int, default=256, help="Batch size for data loading"
    )

    args = parser.parse_args()

    main(args)
