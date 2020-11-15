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
from pysrc.data import mnist, mnistm

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
    pdb.set_trace()
    # Get data loaders
    # MNIST is there in all cases
    mnist_train = mnist.get_mnist(options, train=True)
    mnist_eval = mnist.get_mnist(options, train=False)
    dataset1 = dataset.TransferDataset(name='mnist',
                                       train=mnist_train,
                                       evalset=mnist_eval)

    if options.target == 'mnistm' or options.source == 'mnistm':
        mnistm_train = mnistm.get_mnistm(options, train=True)
        mnistm_eval = mnistm.get_mnistm(options, train=False)
        dataset2 = dataset.TransferDataset(name='mnistm',
                                           train=mnistm_train,
                                           evalset=mnistm_eval)
    elif options.source != 'mnistm' and options.target != 'mnistm':
        raise NotImplementedError

    source = dataset1 if options.source == 'mnist' else dataset2
    target = dataset2 if options.target == 'mnistm' else dataset1

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
