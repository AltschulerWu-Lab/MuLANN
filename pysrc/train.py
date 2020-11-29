import torch
import torch.nn as nn

from pysrc.transfer_options import vanilla_loss, dann_loss, mulann_loss
from pysrc.utils import utils

import torch.backends.cudnn as cudnn
cudnn.benchmark = True
MODEL_FILENAME = "{model}_{method}_{source}_{target}_{{}}.pt"


def train(model, options, source, target, logger):
    model_filename = MODEL_FILENAME.format(model='small', method=options.domain_method,
                                           source=options.source, target=options.target)
    ####################
    # Optimizer #
    optimizer = utils.get_optimizer(options, model)

    ####################
    # Criteria #
    class_criterion = nn.CrossEntropyLoss()
    domain_criterion = nn.BCELoss()
    info_criterion = domain_criterion

    ####################
    # Training #
    global_step = 0
    device = options.device
    model.train()

    # Domain labels
    source_dom_label = torch.zeros(options.batchsize, 1).to(device)  # source 0
    target_dom_label = torch.ones(options.batchsize, 1).to(device)  # target 1

    def train_epoch(data, curr_epoch, len_, total_step):

        if options.domain_method == 'None':
            propagate_func = vanilla_loss
        elif options.domain_method == 'DANN':
            propagate_func = dann_loss
        elif options.domain_method == 'MuLANN':
            propagate_func = mulann_loss
        else:
            raise ValueError('Not implemented')

        for step, ((source_images, source_labels),
                   (lab_target_images, target_labels),
                   (unlab_target_images, _)) in enumerate(data):
            # Lambda du papier DANN
            # p = float(step + epoch * len_dataloader) / options.num_epochs / len_dataloader
            #        alpha = 2. / (1. + np.exp(-10 * p)) - 1
            alpha = options.domain_lambda

            # Move data to the GPU
            source_labels = source_labels.to(device)
            source_images = source_images.to(device)
            lab_target_images = lab_target_images.to(device)
            target_labels = target_labels.to(device)

            # Zero gradients
            optimizer.zero_grad()

            loss, src_class_loss, tgt_class_loss, src_dom_loss, tgt_dom_loss = propagate_func(
                model=model, source_images=source_images, source_labels=source_labels,
                source_dom_label=source_dom_label,
                lab_target_images=lab_target_images, target_labels=target_labels, target_dom_label=target_dom_label,
                unlab_target_images=unlab_target_images,
                class_criterion=class_criterion, domain_criterion=domain_criterion, domain_lambda=alpha, device=device
            )
            # Backward pass
            loss.backward()
            optimizer.step()
            # scheduler.step()
            total_step += 1

            # print step info
            logger.add_scalar('source_class_loss', src_class_loss.item(), total_step)
            logger.add_scalar('target_class_loss', tgt_class_loss.item(), total_step)
            logger.add_scalar('source_dom_loss', src_dom_loss.item(), total_step)
            logger.add_scalar('target_dom_loss', tgt_dom_loss.item(), total_step)
            logger.add_scalar('loss', loss.item(), total_step)

        return src_class_loss, tgt_class_loss, src_dom_loss, tgt_dom_loss, total_step

    for epoch in range(options.num_epochs):
        # zip source and target data pair
        len_dataloader = min(len(source.sup_train), len(target.sup_train), len(target.unsup_train))
        data_zip = zip(source.sup_train, target.sup_train, target.unsup_train)

        source_class_loss, target_class_loss, source_dom_loss, target_dom_loss, global_step = train_epoch(
            data=data_zip, curr_epoch=epoch, len_=len_dataloader, total_step=global_step)

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
