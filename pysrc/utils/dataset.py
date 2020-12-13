from dataclasses import dataclass
from torch.utils.data import DataLoader
from typing import Any


@dataclass
class TransferDataset:
    name: str = ''
    classes: Any = None
    sup_train: DataLoader = None
    unsup_train: DataLoader = None
    sup_val: DataLoader = None
    unsup_val: DataLoader = None
    sup_test: DataLoader = None
    unsup_test: DataLoader = None
