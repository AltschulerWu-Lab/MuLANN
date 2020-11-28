import pdb
import sys
from pathlib import Path
from optparse import OptionParser

sys.path.append(str(Path.cwd()))
from pysrc.utils import parameters
from pysrc.models.large import LargeNet
from pysrc.main import launch_training
from pysrc.data.office_loader import OfficeGetLoader

# Default values correspond to MNIST exp
options = parameters.Params()
options.result_folder = options.result_folder / "office"


if __name__ == '__main__':
    parser = OptionParser(usage="usage: %prog [options]")

    parser.add_option('--gpu', type=int, default=None)
    parser.add_option('--num_workers', type=int, default=None)
    parser.add_option('--fold', type=int, default=None)
    parser.add_option('--eta0', type=int, default=None)
    parser.add_option('--info_zeta', type=float, default=None)
    parser.add_option('--unknown_perc', type=float, default=None)
    parser.add_option('--lrdecay', type=int, default=None)
    parser.add_option('--data_folder', type=str, default=None)
    parser.add_option('--result_folder', type=str, default=None)
    parser.add_option('--domain_lambda', type=float, default=None)
    parser.add_option('--domain_method', type=str, default=None)

    parser.add_option('--source', type=str, default="webcam")
    parser.add_option('--target', type=str, default="amazon")

    (_, args) = parser.parse_args()
    for k, v in parser.values.__dict__.items():
        if v:
            options.__setattr__(k, v)
    launch_training(options, LargeNet, OfficeGetLoader)
