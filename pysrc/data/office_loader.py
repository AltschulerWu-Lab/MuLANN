import torch
from torchvision import transforms
from pysrc.data.usual.office import Office31
from pysrc.data.loader_getter import GetLoader


class OfficeGetLoader(GetLoader):
    dataset = 'office'

    def __init__(self, options, dataset_name, role):
        if dataset_name in {'amazon', 'dslr', 'webcam'}:
            # We do not need the list of semi-sup classes, it's in the train/test split files
            self.domain = dataset_name
            self.role = role

        else:
            raise NotImplementedError

        super().__init__(options)

    def _get_dataset(self, **kwargs):
        """Get Office datasets loader."""
        # image pre-processing
        pre_process = transforms.Compose([transforms.Resize(227),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                               std=(0.229, 0.224, 0.225))
                                          ])

        # datasets and data_loader
        office_dataset = Office31(
            root=self.options.data_folder,
            domain=self.domain,
            role=self.role,
            split=self.options.fold,
            train=kwargs['train'],
            transform=pre_process
        )

        return office_dataset

    def get_semisup(self, **kwargs):
        train = kwargs['train']
        if train:
            # Here train is equivalent to labelled
            return self.get_vanilla(**kwargs)

        # So case where train=False, which means it is unlabelled data
        if self.options.fully_transductive:
            unsup_train = self.get_vanilla(**kwargs)
            unsup_test = unsup_train
            return unsup_train, unsup_test

        else:
            # Loading dataset
            dataset = self._get_dataset(train=train)

            return self._split_in_halves(dataset)
