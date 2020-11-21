import pdb
import sys
import torch
from pathlib import Path
from optparse import OptionParser
from torch.utils.tensorboard import SummaryWriter

sys.path.append("/home/lalil0u/workspace/MuLANN/")
from pysrc.utils import parameters, utils, dataset
from pysrc.train import train
from pysrc.models.small import SmallNet
from pysrc.data import loader_getter, mnistm

# Default values correspond to MNIST exp
options = parameters.Params()
options.result_folder = Path("/home/lalil0u/workspace/MuLANN/results") / "digits"


def launch_train():
    # Set random seed
    utils.init_random_seed(options.seed)

    # Set device
    options.device = torch.device(f'cuda:{options.gpu_id}') if torch.cuda.is_available() \
        else torch.device('cpu')

    # Get model, weights initialized already
    model = SmallNet(options)
    # Get data loaders
    # MNIST
    mnist_getter = loader_getter.GetLoader(options, 'mnist')
    mnistm_getter = loader_getter.GetLoader(options, 'mnistm')
    if options.source == 'mnist':
        get_source = mnist_getter
        get_target = mnistm_getter
    elif options.source == 'mnistm':
        get_source = mnist_getter
        get_target = mnistm_getter
    else:
        raise NotImplementedError

    source = dataset.TransferDataset(name='mnist',
                                     sup_train=get_source.get_vanilla(train=True),
                                     sup_evalset=get_source.get_vanilla(train=False))

    target = dataset.TransferDataset(name='mnistm',
                                     sup_train=get_target.get_semisup(train=True, labelled=True),
                                     sup_evalset=get_target.get_semisup(train=False, labelled=True),
                                     unsup_train=get_target.get_semisup(train=True, labelled=False),
                                     unsup_evalset=get_target.get_semisup(train=False, labelled=False))

    pdb.set_trace()
    # Train
    train(model, options, source, target, SummaryWriter())


if __name__ == '__main__':
    parser = OptionParser(usage="usage: %prog [options]")

    parser.add_option('--gpu', type=int, default=None)
    parser.add_option('--num_workers', type=int, default=None)
    parser.add_option('--seed', type=int, default=None)
    parser.add_option('--eta0', type=int, default=None)
    parser.add_option('--info_zeta', type=int, default=None)
    parser.add_option('--lrdecay', type=int, default=None)
    parser.add_option('--domain_lambda', type=int, default=None)
    parser.add_option('--domain_method', type=str, default=None)

    parser.add_option('--source', type=str, default="mnist")
    parser.add_option('--target', type=str, default="mnistm")

    (_, args) = parser.parse_args()
    for k, v in parser.values.__dict__.items():
        if v:
            options.__setattr__(k, v)
    launch_train()
