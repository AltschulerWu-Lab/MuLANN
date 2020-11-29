
from random import randint


def vanilla_loss(model, source_images, source_labels, class_criterion, alpha, **kwargs):
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
                unlab_target_images,
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
    labtgt_class_output, labtgt_domain_output, labtgt_features = model.forward(input_=lab_target_images,
                                                                               lambda_=domain_lambda,
                                                                               skip_binaries=False)
    info_predictions = model.info_predictor(labtgt_features[:knowledge_batch], param_value=info_zeta)
    labtgt_info_loss = info_criterion(info_predictions, source_dom_label[:knowledge_batch])

    unlab_target_images = unlab_target_images.to(device)
    unlabtgt_class_output, unlabtgt_domain_output, unlabtgt_features = model.forward(input_=unlab_target_images,
                                                                                     lambda_=domain_lambda,
                                                                                     skip_binaries=False)
    info_predictions = model.info_predictor(unlabtgt_features[entropy_mask], param_value=info_zeta)
    unlabtgt_info_loss = info_criterion(info_predictions, target_dom_label[:knowledge_batch])

    # Every other time, I'm using the unlab images to train the domain disc
    r = bool(randint(0, 1))
    tgt_domain_output = unlabtgt_domain_output if r else labtgt_domain_output

    target_class_loss = class_criterion(labtgt_class_output, target_labels)
    target_dom_loss = domain_criterion(tgt_domain_output, target_dom_label)

    # Not putting lambda here, it's taken care of in the middle
    loss = source_class_loss + source_dom_loss + target_dom_loss + target_class_loss \
           + labtgt_info_loss + unlabtgt_info_loss

    return loss, source_class_loss, target_class_loss, source_dom_loss, target_dom_loss
