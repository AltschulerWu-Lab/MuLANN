"""
Dataset setting and data loader for MNIST.
Adapted from @wogong
"""

import torch
from torchvision import datasets, transforms
import os


def get_mnist(batch_size, train, num_workers):
    """Get MNIST datasets loader."""
    # image pre-processing
    pre_process = transforms.Compose([transforms.Resize(32), # 32 is expected
                                      transforms.ToTensor(),
                                      transforms.Normalize(
                                          mean=(0.5, 0.5, 0.5),
                                          std=(0.5, 0.5, 0.5)
                                      )])

    # datasets and data loader
    mnist_dataset = datasets.MNIST(root='./data',
                                   train=train, download=train,
                                   transform=pre_process)

    mnist_data_loader = torch.utils.data.DataLoader(
        dataset=mnist_dataset,
        batch_size=batch_size,
        shuffle=train,
        # Drops last non-full mini-batch
        drop_last=True,
        num_workers=num_workers)

    return mnist_data_loader
