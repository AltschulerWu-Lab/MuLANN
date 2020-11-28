from torchvision import datasets, transforms
from pysrc.data.usual.mnistm import MNISTM
from pysrc.data.loader_getter import GetLoader


class DigitsGetLoader(GetLoader):
    dataset = 'digits'

    def __init__(self, options, dataset_name, **kwargs):
        if dataset_name == 'mnist':
            self._get_dataset = self._get_dataset_mnist

        elif dataset_name == 'mnistm':
            self._get_dataset = self._get_dataset_mnistm

        super().__init__(options)

    def _get_dataset_mnist(self, train, **kwargs):
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

    def _get_dataset_mnistm(self, train, **kwargs):
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

    def get_semisup(self, **kwargs):
        train = kwargs['train']
        labelled = kwargs['labelled']

        # Loading dataset
        dataset = self._get_dataset(train=train)
        subset = self._get_class_subset(dataset, labelled)
        # I can load the unlabelled test dataset for use during training
        # OR I can load the labelled train dataset for use during training
        # OR I can load unlab train to be split in halves for val and test in fully transductive setting,
        # OR I can load the unlabelled test dataset for val and test in non fully transductive setting
        if 'use' not in kwargs or ('use' in kwargs and kwargs['use'] == 'train'):
            return self._get_dataloader(subset, shuffle=True)

        elif kwargs['use'] == 'val_test':
            return self._split_in_halves(subset)
        else:
            raise ValueError('Use should be either train, or val_test.')
