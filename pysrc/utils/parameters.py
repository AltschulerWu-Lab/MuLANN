from dataclasses import dataclass
from typing import Any
from pathlib import Path


# Here I define parameters by default
@dataclass
class Params:
    # DATA
    dataset: str = 'DIGITS'
    inputsize: int = 32
    nchannelsin: int = 3
    nclasses: int = 10
    cell_line: str = 'Cells'
    image_div: int = 4
    num_patch_per_im: int = 10

    # MODEL
    model: str = 'small'

    # GENERAL
    gpu_id: int = 0
    device: Any = None
    seed: int = 42
    num_workers: int = 2
    testingBatch: int = 320
    result_folder: Any = None
    data_folder: Any = Path("/home/lalil0u/workspace/MuLANN/data")

    # OPTIM
    batchsize: int = 128
    train_setting: int = 0 # TODO find the train_setting file corresponding to that train_setting number...
    indiv_lr: bool = False
    eta0: float = 0.001
    lambda_schedule_gamma: float = 10
    gamma: float = 0.001
    beta: float = 0.75
    weightDecay: float = 0
    lr_decay: Any = False
    shuffling_iter: int = 5
    maxiter: int = 400001
    plotinterval: int = 10000
    statinterval: int = 100000

    # TRANSFER SPECIFICS
    source: Any = ''
    target: Any = ''
    domain_adaptation: bool = False
    domain_method: Any = None
    domain_lambda: float = 0
    info_zeta: float = None
    proba: float = 0.5
    common_class_percentage: int = 100
    num_common_labelled_classes: int = 7
    target_unlabelled_presence: bool = True
    fully_transductive: bool = True
    target_supervision_level: int = 50
    iterative_labeling: str = 'entropy'
    unknown_perc: float = 0.7
