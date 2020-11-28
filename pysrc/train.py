import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from random import randint
from pysrc.utils import utils

import torch.backends.cudnn as cudnn
cudnn.benchmark = True
MODEL_FILENAME = "{model}_{method}_{source}_{target}_{{}}.pt"


def train(model, options, source, target, logger):
    model_filename = MODEL_FILENAME.format(model='small', method=options.domain_method,
                                           source=options.source, target=options.target)
    ####################
    # Optimizer #

    if 'office' not in options.source:
        print("training non-office task")
        # Setting function for lr evolution
        # lr_func = utils.adjust_learning_rate
        optimizer = optim.SGD(model.parameters(),
                              lr=options.eta0,
                              momentum=options.momentum,
                              weight_decay=options.weight_decay)

    else:
        print("training office task")
        parameter_list = [{
            "params": model.features.parameters(),
            "lr": 0.001
        }, {
            "params": model.fc.parameters(),
            "lr": 0.001
        }, {
            "params": model.bottleneck.parameters()
        }, {
            "params": model.classifier.parameters()
        }, {
            "params": model.discriminator.parameters()
        }]
        optimizer = optim.SGD(parameter_list, lr=0.01, momentum=0.9)

        # lr_func = utils.adjust_learning_rate_office
    ####################
    # Criteria #
    criterion = nn.CrossEntropyLoss()
    binary = nn.BCELoss()

    ####################
    # Training #
    global_step = 0
    device = options.device
    model.train()

    if options.domain_method == 'None' or options.domain_method == 'DANN':
        train_func = train_epoch

    for epoch in range(options.num_epochs):
        # zip source and target data pair
        len_dataloader = min(len(source.sup_train), len(target.sup_train), len(target.unsup_train))
        data_zip = zip(source.sup_train, target.sup_train, target.unsup_train)

        source_class_loss, target_class_loss, source_dom_loss, target_dom_loss, global_step = train_func(
            options=options, data_zip=data_zip, model=model, criterion=criterion, binary=binary,
            optimizer=optimizer, epoch=epoch, len_dataloader=len_dataloader, global_step=global_step,
            logger=logger
        )

        if (epoch % options.plotinterval) == 0:
            info = (f"Epoch [{epoch + 1:4d}/{options.num_epochs}]: "
                    f"source_class_loss={source_class_loss.data.item():.6f}, "
                    )
            if options.domain_method == "DANN":
                info += (f"target_class_loss={target_class_loss.data.item():.6f}, "
                         f"source_dom_loss={source_dom_loss.data.item():.6f}, "
                         f"target_dom_loss={target_dom_loss.data.item():.6f}, ")
            print(info)

        # eval model
        if (epoch % options.statinterval) == 0:
            tgt_test_loss, tgt_acc, tgt_acc_domain = test(model, target.unsup_test, device, options=options,
                                                          flag='target test')
            logger.add_scalar('tgt_test_loss', tgt_test_loss, global_step)
            logger.add_scalar('tgt_test_acc', tgt_acc, global_step)
            logger.add_scalar('tgt_test_acc_domain', tgt_acc_domain, global_step)
            if target.unsup_val:
                tgt_val_loss, tgt_acc, tgt_acc_domain = test(model, target.unsup_val, device, options=options,
                                                             flag='target val')
                logger.add_scalar('tgt_val_loss', tgt_val_loss, global_step)
                logger.add_scalar('tgt_val_acc', tgt_acc, global_step)
                logger.add_scalar('tgt_val_acc_domain', tgt_acc_domain, global_step)
            if source.sup_test:
                src_test_loss, src_acc, src_acc_domain = test(model, source.sup_test, device, options=options,
                                                              flag='source test')
                logger.add_scalar('src_test_loss', src_test_loss, global_step)
                logger.add_scalar('src_acc', src_acc, global_step)
                logger.add_scalar('src_acc_domain', src_acc_domain, global_step)

    # save final model
    utils.save_model(model, options.result_folder, model_filename.format('final'))

    return model


def train_epoch(options, data_zip, model, criterion, binary, optimizer,
                epoch, len_dataloader, global_step, logger):
    device = options.device
    # Domain labels
    source_dom_label = torch.zeros(options.batchsize, 1).to(device)  # source 0
    target_dom_label = torch.ones(options.batchsize, 1).to(device)  # target 1
    # Default loss values
    source_dom_loss = torch.Tensor(1).fill_(-1)
    target_dom_loss = torch.Tensor(1).fill_(-1)
    target_class_loss = torch.Tensor(1).fill_(-1)

    for step, ((source_images, source_labels),
               (lab_target_images, target_labels),
               (unlab_target_images, _)) in enumerate(data_zip):

        p = float(step + epoch * len_dataloader) / options.num_epochs / len_dataloader
        # Lambda du papier DANN
        alpha = options.domain_lambda
#        alpha = 2. / (1. + np.exp(-10 * p)) - 1

        # Move data to the GPU
        source_labels = source_labels.to(device)
        source_images = source_images.to(device)
        lab_target_images = lab_target_images.to(device)
        target_labels = target_labels.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Fwd source
        src_class_output, src_domain_output, _ = model.forward(input_=source_images,
                                                               param_value=alpha,
                                                               skip_binaries=False)
        # Losses
        source_class_loss = criterion(src_class_output, source_labels)
        loss = source_class_loss
        if options.domain_method == 'DANN':
            # Fwd target - semi-sup so only half classes are here.
            # Other half is in the eval set
            r = bool(randint(0, 1))
            tgt_class_output, tgt_domain_output, _ = model.forward(input_=lab_target_images,
                                                                   param_value=alpha,
                                                                   skip_binaries=r)
            # Every other time, I'm using the unlab images to train the domain disc
            if r:
                unlab_target_images = unlab_target_images.to(device)
                _, tgt_domain_output, _ = model.forward(input_=unlab_target_images,
                                                        param_value=alpha,
                                                        skip_binaries=not r)

            source_dom_loss = binary(src_domain_output, source_dom_label)
            target_class_loss = criterion(tgt_class_output, target_labels)
            target_dom_loss = binary(tgt_domain_output, target_dom_label)

            # Not putting lambda here, it's taken care of in the middle
            loss = source_class_loss + source_dom_loss + target_dom_loss + target_class_loss

        # Backward pass
        loss.backward()
        optimizer.step()

        global_step += 1

        # print step info
        logger.add_scalar('source_class_loss', source_class_loss.item(), global_step)
        logger.add_scalar('source_dom_loss', source_dom_loss.item(), global_step)
        logger.add_scalar('target_dom_loss', target_dom_loss.item(), global_step)
        logger.add_scalar('loss', loss.item(), global_step)

    return source_class_loss, target_class_loss, source_dom_loss, target_dom_loss, global_step


def test(model, data_loader, device, options, flag):
    """Evaluate model for dataset."""
    # set eval state for Dropout and BN layers
    model.eval()

    # init loss and accuracy
    loss_ = 0.0
    acc_ = 0.0
    acc_domain_ = 0.0
    n_total = 0

    # set loss function
    criterion = nn.CrossEntropyLoss()

    # evaluate network
    with torch.no_grad():
        for (images, labels) in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            size = len(labels)
            labels_domain = torch.ones(size).long().to(device) if 'target' in flag else \
                torch.zeros(size).long().to(device)

            preds, domain_preds, _ = model.forward(input_=images, skip_binaries=False)

            loss_ += criterion(preds, labels).item()

            pred_cls = preds.data.max(1)[1]
            acc_ += pred_cls.eq(labels.data).sum().item()
            if options.domain_method == 'DANN':
                pred_domain = domain_preds.data.max(1)[1]
                acc_domain_ += pred_domain.eq(labels_domain.data).sum().item()
            n_total += size

    loss = loss_ / n_total
    acc = acc_ / n_total
    acc_domain = acc_domain_ / n_total

    print(f"Avg {flag} Loss = {loss:.6f}, Avg Accuracy = {acc:.2%}, {acc_}/{n_total}, "
          f"Avg Domain Accuracy = {acc_domain:2%}")
    model.train()
    return loss, acc, acc_domain
