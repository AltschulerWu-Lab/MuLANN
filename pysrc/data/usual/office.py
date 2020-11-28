"""
From https://github.com/thuml/Transfer-Learning-Library under MIT License
Copyright (c) 2018-2020 JunguangJiang
"""
from pathlib import Path
from typing import Optional
from pysrc.data.image_list import ImageList


class Office31(ImageList):
    """Office31 Dataset.
    Parameters:
        - **root** (str): Root directory of dataset
        - **task** (str): The task (domain) to create dataset. Choices include ``'A'``: amazon, \
            ``'D'``: dslr and ``'W'``: webcam.
        - **download** (bool, optional): If true, downloads the dataset from the internet and puts it \
            in root directory. If dataset is already downloaded, it is not downloaded again.
        - **transform** (callable, optional): A function/transform that  takes in an PIL image and returns a \
            transformed version. E.g, ``transforms.RandomCrop``.
        - **target_transform** (callable, optional): A function/transform that takes in the target and transforms it.
    .. note:: In `root`, there will exist following files after downloading.
        ::
            amazon/
                images/
                    backpack/
                        *.jpg
                        ...
            dslr/
            webcam/
            image_list/
                amazon.txt
                dslr.txt
                webcam.txt
    """
    download_list = [
        # Available in the metadata folder of this repo
        ("image_list", ),
        # Available at https://drive.google.com/file/d/0B4IapRTv9pJ1WGZVd1VDMmhwdlE
        ("images", ),
    ]
    SPLIT_FILE = {
        'source': 'split_{:02}.txt',
        'target': 'diff_category-split_{:02}.txt'
    }
    DOMAINS = {'amazon', 'dslr', 'webcam'}
    CLASSES = ['back_pack', 'bike', 'bike_helmet', 'bookcase', 'bottle', 'calculator', 'desk_chair', 'desk_lamp',
               'desktop_computer', 'file_cabinet', 'headphones', 'keyboard', 'laptop_computer', 'letter_tray',
               'mobile_phone', 'monitor', 'mouse', 'mug', 'paper_notebook', 'pen', 'phone', 'printer', 'projector',
               'punchers', 'ring_binder', 'ruler', 'scissors', 'speaker', 'stapler', 'tape_dispenser', 'trash_can']

    def __init__(self, root: str,
                 domain: str,
                 split: int,
                 role: str,
                 train: bool,
                 **kwargs):
        # Data root
        self.root = Path(root) / 'office'

        # Checking domain exists, and data is downloaded already
        assert domain in self.DOMAINS
        assert (self.root / domain).exists()

        self.filename = self.SPLIT_FILE[role].format(split)

        data_list_file = self._get_metadata(domain, role, train)
        super().__init__(self.root, Office31.CLASSES, data_list_file=data_list_file, **kwargs)

    def _get_metadata(self, domain, role, train):
        kw = 'train' if train else 'test'
        metadata_folder = Path.cwd() / 'metadata' / 'train_test_sets' / 'office' / domain / role / kw

        return metadata_folder / self.filename
