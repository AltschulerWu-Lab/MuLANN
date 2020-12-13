import sys
import torch
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

sys.path.append(str(Path.cwd()))
from pysrc.utils import utils, dataset
from pysrc.train import train


def launch_training(options, model_class, data_getter):
    # Set random seed
    utils.init_random_seed(options.seed)

    # Set device
    options.device = torch.device(f'cuda:{options.gpu_id}') if torch.cuda.is_available() \
        else torch.device('cpu')

    # Get model, weights initialized already
    model = model_class(options)
    # Move model to device
    model.to(options.device)

    # Get data loaders
    # SOURCE
    get_source = data_getter(options, dataset_name=options.source, role='source')
    # TARGET
    get_target = data_getter(options, dataset_name=options.target, role='target')

    source = dataset.TransferDataset(name=options.source,
                                     sup_train=get_source.get_vanilla(train=True, shuffle=True),
                                     sup_test=get_source.get_vanilla(train=False, shuffle=False))

    if get_source.dataset == 'digits':
        # NB: In the paper, unlabelled data used during training
        # came from the test split (MNIST data, Office data)
        unsup_train = get_target.get_semisup(train=True, labelled=False, use='train')
        # Fully transductive setting: predicting on the same images than training
        unsup_val, unsup_test = get_target.get_semisup(train=options.fully_transductive,
                                                       labelled=False, use='val_test')
        # Labelled training set
        sup_train = get_target.get_semisup(train=True, labelled=True)
        target = dataset.TransferDataset(name=options.target,
                                         classes=get_target.get_classes(sup_train),
                                         sup_train=sup_train,
                                         sup_test=get_target.get_semisup(train=False, labelled=True),
                                         unsup_train=unsup_train,
                                         unsup_val=unsup_val,
                                         unsup_test=unsup_test)

    elif get_source.dataset == 'office':
        unsup_train, unsup_test = get_target.get_semisup(train=False, shuffle=True)

        # Labelled training set
        sup_train = get_target.get_semisup(train=True, shuffle=True)
        target = dataset.TransferDataset(name=options.target,
                                         classes=get_target.get_classes(sup_train),
                                         sup_train=sup_train,
                                      #   sup_test=get_target.get_semisup(train=False, labelled=True),
                                         unsup_train=unsup_train,
                                         unsup_test=unsup_test)
    else:
        raise NotImplementedError(f'Dataset {get_source.dataset} not yet implemented.')

    # Train
    print(model)
    print(options)
    train(model, options, source, target, SummaryWriter())
