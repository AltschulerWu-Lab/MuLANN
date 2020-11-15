import random
from pathlib import Path
import torch


def init_random_seed(manual_seed=None):
    """
    Sets the random seed everywhere.

    :param manual_seed:
    :return:
    """
    seed = random.randint(1, 10000) if manual_seed is None else manual_seed

    print(f"Random seed: {seed}")
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_model(net, model_root, filename):
    """
    Saves the model

    :param net:
    :param model_root:
    :param filename:
    :return:
    """
    dir_ = Path(model_root)
    if not dir_.exists():
        dir_.mkdir()

    file_ = dir_ / filename
    torch.save(net.state_dict(), file_)
    print(f"Saved to: {file_}")


def adjust_learning_rate(optimizer, p):
    lr_0 = 0.01
    alpha = 10
    beta = 0.75
    lr = lr_0 / (1 + alpha * p)**beta
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def adjust_learning_rate_office(optimizer, p):
    lr_0 = 0.001
    alpha = 10
    beta = 0.75
    lr = lr_0 / (1 + alpha * p)**beta
    for param_group in optimizer.param_groups[:2]:
        param_group['lr'] = lr
    for param_group in optimizer.param_groups[2:]:
        param_group['lr'] = 10 * lr
    return lr
