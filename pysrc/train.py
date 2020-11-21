import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from random import randint
from pysrc.utils import utils

import torch.backends.cudnn as cudnn
cudnn.benchmark = True
MODEL_FILENAME = "{model}_{source}_{target}_{{}}.pt"


def train(model, options, source, target, logger):
    model_filename = MODEL_FILENAME.format(model='dann', source=options.source, target=options.target)
    ####################
    # Optimizer #

    if 'office' not in options.source:
        print("training non-office task")
        # Setting function for lr evolution
        lr_func = utils.adjust_learning_rate
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

        lr_func = utils.adjust_learning_rate_office
    ####################
    # Criteria #
    criterion = nn.CrossEntropyLoss()
    binary = nn.BCELoss()

    ####################
    # Training #
    global_step = 0
    device = options.device
    model.train()
    # Domain labels
    label_src = torch.zeros(options.batchsize).long().to(device)  # source 0
    label_tgt = torch.ones(options.batchsize).long().to(device)  # target 1

    for epoch in range(options.num_epochs):
        # zip source and target data pair
        len_dataloader = min(len(source.sup_train), len(target.sup_train), len(target.unsup_train))
        data_zip = zip(source.sup_train, target.sup_train, target.unsup_train)

        for step, ((source_images, source_labels),
                   (lab_target_images, target_labels),
                   (unlab_target_images, _)) in enumerate(data_zip):

            p = float(step + epoch * len_dataloader) / options.num_epochs / len_dataloader
            # Lambda du papier DANN
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            lr = lr_func(optimizer, p)
            logger.add_scalar('lr', lr, global_step)

            # Move data to the GPU
            source_labels = source_labels.to(device)
            source_images = source_images.to(device)
            lab_target_images = lab_target_images.to(device)
            target_labels = target_labels.to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Fwd source
            src_class_output, src_domain_output, _ = model.forward(input_data=source_images, param_value=alpha,
                                                                   skip_binaries=False)

            # Fwd target - semi-sup so only half classes are here.
            # Other half is in the eval set
            r = bool(randint(0, 1))
            tgt_class_output, tgt_domain_output, _ = model.forward(input_data=lab_target_images,
                                                                   param_value=alpha, skip_binaries=r)
            # Every other time, I'm using the unlab images to train the domain disc
            if r:
                unlab_target_images = unlab_target_images.to(device)
                _, tgt_domain_output, _ = model.forward(input_data=unlab_target_images,
                                                        param_value=alpha, skip_binaries=not r)

            # Losses
            src_loss_class = criterion(src_class_output, source_labels)
            src_loss_domain = binary(src_domain_output, label_src)
            tgt_loss_class = criterion(tgt_class_output, target_labels)
            tgt_loss_domain = binary(tgt_domain_output, label_tgt)

            # Not putting lambda here, it's taken care of in the middle
            loss = src_loss_class + src_loss_domain + tgt_loss_domain + tgt_loss_class

            # Backward pass
            loss.backward()
            optimizer.step()

            global_step += 1

            # print step info
            logger.add_scalar('src_loss_class', src_loss_class.item(), global_step)
            logger.add_scalar('src_loss_domain', src_loss_domain.item(), global_step)
            logger.add_scalar('tgt_loss_domain', tgt_loss_domain.item(), global_step)
            logger.add_scalar('loss', loss.item(), global_step)

            if ((step + 1) % options.plotinterval) == 0:
                print(f"Epoch [{epoch + 1:4d}/{options.num_epochs}] Step [{step + 1:2d}/{len_dataloader}]: "
                      f"src_loss_class={src_loss_class.data.item():.6f}, "
                      f"src_loss_domain={src_loss_domain.data.item():.6f}, "
                      f"tgt_loss_domain={tgt_loss_domain.data.item():.6f}, "
                      f"loss={loss.data.item():.6f}")

        # eval model
        if ((epoch + 1) % options.statinterval) == 0:
            tgt_test_loss, tgt_acc, tgt_acc_domain = test(model, target.unsup_evalset, device, flag='target')
            src_test_loss, src_acc, src_acc_domain = test(model, source.sup_evalset, device, flag='source')
            logger.add_scalar('src_test_loss', src_test_loss, global_step)
            logger.add_scalar('src_acc', src_acc, global_step)
            logger.add_scalar('src_acc_domain', src_acc_domain, global_step)
            logger.add_scalar('tgt_test_loss', tgt_test_loss, global_step)
            logger.add_scalar('tgt_acc', tgt_acc, global_step)
            logger.add_scalar('tgt_acc_domain', tgt_acc_domain, global_step)

        # save model parameters
        if ((epoch + 1) % options.save_step) == 0:
            utils.save_model(model, options.model_root, model_filename.format(epoch + 1))

    # save final model
    utils.save_model(model, options.model_root, model_filename.format('final'))

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
    with torch.no_grad():
        for (images, labels) in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            size = len(labels)
            labels_domain = torch.ones(size).long().to(device) if flag == 'target' else \
                torch.zeros(size).long().to(device)

            preds, domain_preds = model.forward(images)

            loss_ += criterion(preds, labels).item()

            pred_cls = preds.data.max(1)[1]
            pred_domain = domain_preds.data.max(1)[1]
            acc_ += pred_cls.eq(labels.data).sum().item()
            acc_domain_ += pred_domain.eq(labels_domain.data).sum().item()
            n_total += size

    loss = loss_ / n_total
    acc = acc_ / n_total
    acc_domain = acc_domain_ / n_total

    print(f"Avg Loss = {loss:.6f}, Avg Accuracy = {acc:.2%}, {acc_}/{n_total}, Avg Domain Accuracy = {acc_domain:2%}")
    model.train()
    return loss, acc, acc_domain
