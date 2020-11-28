from dataclasses import dataclass
from torch.utils.data import DataLoader


@dataclass
class TransferDataset:
    name: str = ''
    sup_train: DataLoader = None
    unsup_train: DataLoader = None
    sup_val: DataLoader = None
    unsup_val: DataLoader = None
    sup_test: DataLoader = None
    unsup_test: DataLoader = None
