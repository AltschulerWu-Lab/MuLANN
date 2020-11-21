from dataclasses import dataclass
from typing import Any


@dataclass
class TransferDataset:
    name: str = ''
    sup_train: Any = None
    unsup_train: Any = None
    sup_evalset: Any = None
    unsup_evalset: Any = None
