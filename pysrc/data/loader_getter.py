import torch
from torch.utils.data.distributed import DistributedSampler
from pysrc.utils.parameters import semi_sup_classes
import numpy as np
from abc import abstractmethod


class GetLoader(object):
    dataset = ''
    num_classes = 0

    def __init__(self, options):
        self.options = options
        self.semi_sup_classes = semi_sup_classes.get(self.dataset)

    @abstractmethod
    def get_semisup(self, **kwargs):
        pass

    @abstractmethod
    def _get_dataset(self, **kwargs):
        pass

    def get_vanilla(self, **kwargs):
        dataset = self._get_dataset(**kwargs)

        return self._get_dataloader(dataset, kwargs['shuffle'])

    def _split_in_halves(self, dataset):
        val_sampler = DistributedSampler(dataset, num_replicas=2, rank=0, shuffle=False, drop_last=False)
        test_sampler = DistributedSampler(dataset, num_replicas=2, rank=1, shuffle=False, drop_last=False)
        return self._get_dataloader(dataset=dataset, shuffle=False, sampler=val_sampler), \
               self._get_dataloader(dataset=dataset, shuffle=False, sampler=test_sampler)

    def _get_class_subset(self, dataset, labelled):
        labels = torch.tensor(dataset.targets)
        label_idx = [label.item() in self.semi_sup_classes[labelled] for label in labels]

        return torch.utils.data.dataset.Subset(dataset, np.where(label_idx)[0])

    def _get_dataloader(self, dataset, shuffle, sampler=None):
        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.options.batchsize,
            shuffle=shuffle,
            sampler=sampler,
            # Drops last non-full mini-batch
            drop_last=True,
            num_workers=self.options.num_workers)

    def get_classes(self, loader):
        l = []
        for _, labels in loader:
            l.extend(labels)

        counts = np.bincount(l, minlength=self.num_classes)
        return counts > 0
