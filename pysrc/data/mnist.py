"""
Dataset setting and data loader for MNIST.
Adapted from @wogong 2018, MIT license
"""

import torch
from torchvision import datasets, transforms
import os


def get_mnist(options, train):
    """Get MNIST datasets loader."""
    # image pre-processing
    pre_process = transforms.Compose([transforms.Resize(options.inputsize), # 32 is expected
                                      transforms.ToTensor(),
                                      transforms.Normalize(
                                          mean=(0.5, 0.5, 0.5),
                                          std=(0.5, 0.5, 0.5)
                                      )])

    # datasets and data loader
    mnist_dataset = datasets.MNIST(root=options.data_folder,
                                   train=train,
                                   download=train,
                                   transform=pre_process)

    mnist_data_loader = torch.utils.data.DataLoader(
        dataset=mnist_dataset,
        batch_size=options.batchsize,
        shuffle=train,
        # Drops last non-full mini-batch
        drop_last=True,
        num_workers=options.num_workers)

    return mnist_data_loader
