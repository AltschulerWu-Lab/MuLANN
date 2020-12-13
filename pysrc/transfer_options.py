import torch
from random import randint
from pysrc.utils.utils import get_entropy_mask


def vanilla_loss(model, source_images, source_labels, class_criterion, **kwargs):
    # Fwd source
    src_class_output, src_domain_output, _ = model.forward(input_=source_images)
    # Losses
    source_class_loss = class_criterion(src_class_output, source_labels)
    loss = source_class_loss

    return loss, source_class_loss, None, None, None


def dann_loss(model,
              source_images, source_labels, source_dom_label,
              lab_target_images, target_labels, target_dom_label,
              unlab_target_images,
              class_criterion, domain_criterion, domain_lambda,
              device, **kwargs):

    # Fwd source
    src_class_output, src_domain_output, _ = model.forward(input_=source_images,
                                                           lambda_=domain_lambda,
                                                           skip_binaries=False)
    # Losses
    source_class_loss = class_criterion(src_class_output, source_labels)
    # Fwd target - semi-sup so only half classes are here.
    # Other half is in the eval set
    r = bool(randint(0, 1))
    tgt_class_output, tgt_domain_output, _ = model.forward(input_=lab_target_images,
                                                           lambda_=domain_lambda,
                                                           skip_binaries=r)
    # Every other time, I'm using the unlab images to train the domain disc
    if r:
        unlab_target_images = unlab_target_images.to(device)
        _, tgt_domain_output, _ = model.forward(input_=unlab_target_images,
                                                lambda_=domain_lambda,
                                                skip_binaries=not r)

    source_dom_loss = domain_criterion(src_domain_output, source_dom_label)
    target_class_loss = class_criterion(tgt_class_output, target_labels)
    target_dom_loss = domain_criterion(tgt_domain_output, target_dom_label)

    # Not putting lambda here, it's taken care of in the middle
    loss = source_class_loss + source_dom_loss + target_dom_loss + target_class_loss

    return loss, source_class_loss, target_class_loss, source_dom_loss, target_dom_loss


def mulann_loss(model, source_images, source_labels, source_dom_label,
                lab_target_images, target_labels, target_dom_label,
                unlab_target_images, target_classes,
                class_criterion, domain_criterion, domain_lambda,
                info_criterion, info_zeta, knowledge_batch,
                device):

    # Fwd source
    src_class_output, src_domain_output, _ = model.forward(input_=source_images,
                                                           lambda_=domain_lambda,
                                                           skip_binaries=False)
    # Losses
    source_class_loss = class_criterion(src_class_output, source_labels)
    source_dom_loss = domain_criterion(src_domain_output, source_dom_label)

    # Fwd target - semi-sup so only half classes are here.
    # Other half is in the eval set
    labtgt_class_output, labtgt_domain_output, labtgt_info_output = model.forward(input_=lab_target_images,
                                                                                  lambda_=domain_lambda,
                                                                                  zeta=info_zeta,
                                                                                  skip_binaries=False)
    labtgt_info_loss = info_criterion(labtgt_info_output, source_dom_label)
    mask = torch.zeros(*labtgt_info_loss.shape, dtype=torch.bool)
    mask[:knowledge_batch] = True
    labtgt_info_loss = labtgt_info_loss.masked_select(mask).mean()

    unlab_target_images = unlab_target_images.to(device)
    unlabtgt_class_output, unlabtgt_domain_output, unlabtgt_info_output = model.forward(input_=unlab_target_images,
                                                                                        lambda_=domain_lambda,
                                                                                        zeta=info_zeta,
                                                                                        skip_binaries=False)
    unlabtgt_info_loss = info_criterion(unlabtgt_info_output, target_dom_label)
    # Selecting top 'knowledge_batch' wrt entropy, to go through the Known-Unknown Disc
    entropy_mask = get_entropy_mask(unlabtgt_class_output, target_classes, knowledge_batch)
    unlabtgt_info_loss = unlabtgt_info_loss.masked_select(entropy_mask).mean()

    # Every other time, I'm using the unlab images to train the domain disc
    r = bool(randint(0, 1))
    tgt_domain_output = unlabtgt_domain_output if r else labtgt_domain_output

    target_class_loss = class_criterion(labtgt_class_output, target_labels)
    target_dom_loss = domain_criterion(tgt_domain_output, target_dom_label)

    # Not putting lambda here, it's taken care of in the middle
    loss = source_class_loss + source_dom_loss + target_dom_loss + target_class_loss \
           + labtgt_info_loss + unlabtgt_info_loss

    return loss, source_class_loss, target_class_loss, source_dom_loss, target_dom_loss
