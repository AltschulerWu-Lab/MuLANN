from dataclasses import dataclass
from typing import Any


@dataclass
class TransferDataset:
    name: str = ''
    train: Any = None
    evalset: Any = None
