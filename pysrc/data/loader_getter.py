from pathlib import Path

import torch
from torchvision import datasets, transforms
from pysrc.utils.parameters import semi_sup_classes
from pysrc.data.mnistm import MNISTM
import numpy as np


class GetLoader(object):
    def __init__(self, options, dataset_name):
        self.options = options
        if dataset_name == 'mnist':
            self._get_dataset = self._get_dataset_mnist
            self.semi_sup_classes = semi_sup_classes['digits']
        elif dataset_name == 'mnistm':
            self._get_dataset = self._get_dataset_mnistm
            self.semi_sup_classes = semi_sup_classes['digits']
        else:
            raise NotImplementedError

    def get_semisup(self, train, labelled):
        dataset = self._get_dataset(train)

        labels = torch.tensor(dataset.targets)
        label_idx = [label.item() in self.semi_sup_classes[labelled] for label in labels]

        return self._get_dataloader(dataset=torch.utils.data.dataset.Subset(dataset,
                                                                            np.where(label_idx)[0]),
                                    train=train)

    def get_vanilla(self, train):
        dataset = self._get_dataset(train)

        return self._get_dataloader(dataset, train)

    def _get_dataset_office(self, category):
        """Get Office datasets loader."""
        # image pre-processing
        pre_process = transforms.Compose([transforms.Resize(227),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                               std=(0.229, 0.224, 0.225))
                                          ])

        # datasets and data_loader
        office_dataset = datasets.ImageFolder((Path(self.options.data_folder) / 'office' / category / 'images'),
                                              transform=pre_process)

        return office_dataset

    def _get_dataset_mnistm(self, train):
        """Get MNIST_M datasets loader."""
        # image pre-processing
        pre_process = transforms.Compose([transforms.Resize(self.options.inputsize),  # 32 is expected
                                          transforms.ToTensor(),
                                          transforms.Normalize(
                                              mean=(0.5, 0.5, 0.5),
                                              std=(0.5, 0.5, 0.5)
                                          )])

        # datasets and data loader
        mnistm_dataset = MNISTM(root=self.options.data_folder,
                                mnist_root=self.options.data_folder,
                                train=train, download=train,
                                transform=pre_process)

        return mnistm_dataset

    def _get_dataset_mnist(self, train):
        # image pre-processing
        pre_process = transforms.Compose([transforms.Resize(self.options.inputsize), # 32 is expected
                                          transforms.Grayscale(3),
                                          transforms.ToTensor(),
                                          transforms.Normalize(
                                              mean=(0.5, 0.5, 0.5),
                                              std=(0.5, 0.5, 0.5)
                                          )])

        # datasets and data loader
        mnist_dataset = datasets.MNIST(root=self.options.data_folder,
                                       train=train,
                                       download=train,
                                       transform=pre_process)
        return mnist_dataset

    def _get_dataloader(self, dataset, train):
        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.options.batchsize,
            shuffle=train,
            # Drops last non-full mini-batch
            drop_last=True,
            num_workers=self.options.num_workers)
