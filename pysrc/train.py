"""
Train function structure inspired from @wogong 2018, MIT license
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pysrc.data import mnistm
from pysrc.models import small
from pysrc.utils import utils

import torch.backends.cudnn as cudnn
cudnn.benchmark = True


def train(model, options, source, target, logger):
    ####################
    # 1. setup criterion and optimizer #
    ####################

    if not options.finetune_flag:
        print("training non-office task")
        optimizer = optim.SGD(model.parameters(),
                              lr=options.lr,
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

    criterion = nn.CrossEntropyLoss()
    binary = nn.BCELoss()

    ####################
    # 2. train network #
    ####################
    global_step = 0
    device = options.device
    model.train()

    for epoch in range(options.num_epochs):
        # zip source and target data pair
        len_dataloader = min(len(source), len(target))
        for step, ((images_src, class_src), (images_tgt, _)) in enumerate(zip(source, target)):

            p = float(step + epoch * len_dataloader) / options.num_epochs / len_dataloader
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            if options.lr_adjust_flag == 'simple':
                lr = utils.adjust_learning_rate(optimizer, p)
            else:
                lr = utils.adjust_learning_rate_office(optimizer, p)
            logger.add_scalar('lr', lr, global_step)

            # prepare domain label
            size_src = len(images_src)
            size_tgt = len(images_tgt)
            label_src = torch.zeros(size_src).long().to(device)  # source 0
            label_tgt = torch.ones(size_tgt).long().to(device)  # target 1

            # make images variable
            class_src = class_src.to(device)
            images_src = images_src.to(device)
            images_tgt = images_tgt.to(device)

            # zero gradients for optimizer
            optimizer.zero_grad()

            # train on source domain
            src_class_output, src_domain_output = model(input_data=images_src, alpha=alpha)
            src_loss_class = criterion(src_class_output, class_src)
            src_loss_domain = criterion(src_domain_output, label_src)

            # train on target domain
            _, tgt_domain_output = model(input_data=images_tgt, alpha=alpha)
            tgt_loss_domain = criterion(tgt_domain_output, label_tgt)

            loss = src_loss_class + src_loss_domain + tgt_loss_domain
            if options.src_only_flag:
                loss = src_loss_class

            # optimize dann
            loss.backward()
            optimizer.step()

            global_step += 1

            # print step info
            logger.add_scalar('src_loss_class', src_loss_class.item(), global_step)
            logger.add_scalar('src_loss_domain', src_loss_domain.item(), global_step)
            logger.add_scalar('tgt_loss_domain', tgt_loss_domain.item(), global_step)
            logger.add_scalar('loss', loss.item(), global_step)

            if ((step + 1) % options.log_step) == 0:
                print(f"Epoch [{epoch + 1:4d}/{options.num_epochs}] Step [{step + 1:2d}/{len_dataloader}]: "
                      f"src_loss_class={src_loss_class.data.item():.6f}, "
                      f"src_loss_domain={src_loss_domain.data.item():.6f}, "
                      f"tgt_loss_domain={tgt_loss_domain.data.item():.6f}, "
                      f"loss={loss.data.item():.6f}")

        # eval model
        if ((epoch + 1) % options.eval_step) == 0:
            tgt_test_loss, tgt_acc, tgt_acc_domain = test(model, tgt_data_loader_eval, device, flag='target')
            src_test_loss, src_acc, src_acc_domain = test(model, source, device, flag='source')
            logger.add_scalar('src_test_loss', src_test_loss, global_step)
            logger.add_scalar('src_acc', src_acc, global_step)
            logger.add_scalar('src_acc_domain', src_acc_domain, global_step)
            logger.add_scalar('tgt_test_loss', tgt_test_loss, global_step)
            logger.add_scalar('tgt_acc', tgt_acc, global_step)
            logger.add_scalar('tgt_acc_domain', tgt_acc_domain, global_step)

        # save model parameters
        if ((epoch + 1) % options.save_step) == 0:
            utils.save_model(model, options.model_root,
                             options.src_dataset + '-' + options.tgt_dataset + "-dann-{}.pt".format(epoch + 1))

    # save final model
    utils.save_model(model, options.model_root, options.src_dataset + '-' + options.tgt_dataset + "-dann-final.pt")

    return model


def test(model, data_loader, device, flag):
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
    for (images, labels) in data_loader:
        images = images.to(device)
        labels = labels.to(device)
        size = len(labels)
        labels_domain = torch.ones(size).long().to(device) if flag == 'target' else torch.zeros(size).long().to(device)

        preds, domain = model(images, alpha=0)

        loss_ += criterion(preds, labels).item()

        pred_cls = preds.data.max(1)[1]
        pred_domain = domain.data.max(1)[1]
        acc_ += pred_cls.eq(labels.data).sum().item()
        acc_domain_ += pred_domain.eq(labels_domain.data).sum().item()
        n_total += size

    loss = loss_ / n_total
    acc = acc_ / n_total
    acc_domain = acc_domain_ / n_total

    print(f"Avg Loss = {loss:.6f}, Avg Accuracy = {acc:.2%}, {acc_}/{n_total}, Avg Domain Accuracy = {acc_domain:2%}")
    model.train()
    return loss, acc, acc_domain
