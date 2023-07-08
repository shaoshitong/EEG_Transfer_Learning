import torch
import torch.nn.functional as F
import numpy as np
from KD.base_kd import hinton_distillation_one, hinton_distillation_wo_ce
from one_utils import eval, LoggerForSacred, adjust_learning_rate, get_config_var
import cmodels.WideLinear as WideLinear
import  os

save_dir = get_config_var()["SAVE_DIR"]

def grl_multi_target_hinton_train_alt(current_ep, epochs, model, optimizer_da, optimizer_kd, device,

                         source_dataloader, targets_dataloader, T, alpha, beta, gamma, batch_norm, is_cst, is_debug=False,  **kwargs):

    logger = kwargs["logger"]
    if "logger_id" not in kwargs:
        logger_id = ""
    else:
        logger_id = kwargs["logger_id"]

    if batch_norm:
        model.train()

    total_losses = torch.zeros(len(targets_dataloader))
    da_temp_losses = torch.zeros(len(targets_dataloader))
    kd_temp_losses = torch.zeros(len(targets_dataloader))
    kd_target_loss = 0.
    kd_source_loss = 0.

    iter_targets = [0] * len(targets_dataloader)
    for i, d in enumerate(targets_dataloader):
        iter_targets[i] = iter(d)

    iter_source = iter(source_dataloader)#迭代器

    for i in range(1, len(source_dataloader) + 1):

        data_source, label_source = iter_source.next()
        data_source = data_source.to(device)
        label_source = label_source.to(device)

        for ix, it in enumerate(iter_targets):
            try:
                data_target, _ = it.next()
            except StopIteration:
                it = iter(targets_dataloader[ix])
                data_target, _ = it.next()

            if data_target.shape[0] != data_source.shape[0]:
                data_target = data_target[: data_source.shape[0]] #输出第data_source.shape[0]列上的元素
            data_target = data_target.to(device)

            # model_path = r"F:/wsy/研一/科研/Mymodel/MT-MTDA - subject - one - WFE - their/models_one_their/one_model_14_%d.pth" % ix
            # teacher_one_model = WideLinear.WLE(
            #     num_classes=3,
            #     channels=32,
            #     hidden_dim=64,
            #     d_model=100,
            #     grad_rate=0.9,
            #     use_inf=True
            # ).to(device)
            # teacher_one_model.load_state_dict(torch.load(model_path))
            # teacher_one_model.eval()
            # with torch.no_grad():
            #     _, _, _, _, _, _, teacher_source_output_m = teacher_one_model(data_source)
            #     _, _, _, _, _, _, teacher_target_output_m = teacher_one_model(data_target)

            p = float(i + (current_ep -1) * len(source_dataloader)) / epochs / len(source_dataloader)
            delta = 2. / (1. + np.exp(-10 * p)) - 1
            label_source_pred_1, label_source_pred_2, \
            label_source_pred_3, label_source_pred_m, \
            source_loss_adv_1, source_loss_adv_2,\
            source_loss_adv_3 = model(data_source, delta)
            source_loss_cls_1 = F.cross_entropy(F.softmax(label_source_pred_1,dim=1), label_source)  # TODO FLAG1
            source_loss_cls_2 = F.cross_entropy(F.softmax(label_source_pred_2,dim=1), label_source)
            source_loss_cls_3 = F.cross_entropy(F.softmax(label_source_pred_3,dim=1), label_source)
            source_loss_cls_m = F.cross_entropy(F.softmax(label_source_pred_m,dim=1), label_source)
            source_loss_adv = source_loss_adv_1 + source_loss_adv_2 + source_loss_adv_3
            source_loss_cls = source_loss_cls_1 + source_loss_cls_2 + source_loss_cls_3 + source_loss_cls_m

            target_logits_1, target_logits_2, \
            target_logits_3, target_logits_m, \
            target_loss_adv_1, target_loss_adv_2,\
            target_loss_adv_3 = model(data_target, delta, source=False)
            target_loss_adv = target_loss_adv_1 + target_loss_adv_2 + target_loss_adv_3
            loss_adv = source_loss_adv + target_loss_adv

            da_grl_loss = (1 - beta) * (source_loss_cls + gamma * loss_adv) #1-β LDA # TODO += -> =

            source_kd_loss_1 = hinton_distillation_one(label_source_pred_1, label_source_pred_m, label_source, T, alpha).abs()
            source_kd_loss_2 = hinton_distillation_one(label_source_pred_2, label_source_pred_m, label_source, T, alpha).abs()
            source_kd_loss_3 = hinton_distillation_one(label_source_pred_3, label_source_pred_m, label_source, T, alpha).abs()
            # source_teacher_kd_loss = hinton_distillation_one(label_source_pred_m, teacher_source_output_m, label_source,T, alpha).abs()

            source_kd_loss = source_kd_loss_1 + source_kd_loss_2 + source_kd_loss_3
            if is_cst:
                target_kd_loss_1 = hinton_distillation_wo_ce(target_logits_1, target_logits_m, T).abs()
                target_kd_loss_2 = hinton_distillation_wo_ce(target_logits_2, target_logits_m, T).abs()
                target_kd_loss_3 = hinton_distillation_wo_ce(target_logits_3, target_logits_m, T).abs()
                # target_teacher_kd_loss = hinton_distillation_wo_ce(target_logits_m, teacher_target_output_m, T).abs()
                target_kd_loss = (target_kd_loss_1 + target_kd_loss_2 + target_kd_loss_3  )+ alpha * target_loss_adv
                #target_kd_loss = target_kd_loss.data# + alpha * target_loss_adv # TODO w/o alpha*target_loss_adv
            else:
                target_kd_loss_1 = hinton_distillation_wo_ce(target_logits_1, target_logits_m, T).abs()
                target_kd_loss_2 = hinton_distillation_wo_ce(target_logits_2, target_logits_m, T).abs()
                target_kd_loss_3 = hinton_distillation_wo_ce(target_logits_3, target_logits_m, T).abs()
                # target_teacher_kd_loss = hinton_distillation_wo_ce(target_logits_m, teacher_target_output_m, T).abs()
                target_kd_loss =  target_kd_loss_1 + target_kd_loss_2 + target_kd_loss_3

            total_losses = (da_grl_loss + beta * (source_kd_loss + target_kd_loss)).mean()

            da_temp_losses[ix] += total_losses.item()
            optimizer_da.zero_grad()
            total_losses.backward()
            optimizer_da.step()  # May need to have 2 optimizers

        if is_debug:
            break

    del da_grl_loss
    return da_temp_losses / len(source_dataloader)

def grl_multi_target_hinton_alt(init_lr_da, init_lr_kd, device, epochs, T, alpha, gamma, growth_rate, init_beta,
                   source_dloader, targets_dloader, targets_testloader, optimizer_da, optimizer_kd, model,
                   is_scheduler_da=True, is_scheduler_kd=False, scheduler_da=None, scheduler_kd=None, is_debug=False, save_name="", batch_norm=False, is_cst=True, **kwargs):

    logger = kwargs["logger"]
    if "logger_id" not in kwargs:
        logger_id = ""
    else:
        logger_id = kwargs["logger_id"]

    best_acc_1 = 0.
    best_acc_2 = 0.
    best_acc_3 = 0.
    best_acc_m = 0.
    epochs += 1

    for epoch in range(1, epochs):

        beta = init_beta * torch.exp(growth_rate * (epoch - 1))
        beta = beta.to(device)
        if is_scheduler_da:
            new_lr_da = init_lr_da / np.power((1 + 10 * (epoch - 1) / epochs), 0.75) # 10*
            adjust_learning_rate(optimizer_da, new_lr_da)

        if is_scheduler_kd:
            new_lr_kd = init_lr_kd / np.power((1 + 10 * (epoch - 1) / epochs), 0.75) # 10*
            adjust_learning_rate(optimizer_kd, new_lr_kd)

        total_losses = grl_multi_target_hinton_train_alt(epoch, epochs, model, optimizer_da,
                                                            optimizer_kd, device, source_dloader, targets_dloader, T,
                                                            alpha, beta, gamma, batch_norm, is_cst, is_debug, logger=None)

        targets_acc_1 = np.zeros(len(targets_dloader))
        targets_acc_2 = np.zeros(len(targets_dloader))
        targets_acc_3 = np.zeros(len(targets_dloader))
        targets_acc_m = np.zeros(len(targets_dloader))


        for i, d in enumerate(targets_testloader):
            targets_acc_1[i], targets_acc_2[i], targets_acc_3[i], targets_acc_m[i] = eval(model, device, d, is_debug)

        total_target_acc_1 = targets_acc_1.mean()
        total_target_acc_2 = targets_acc_2.mean()
        total_target_acc_3 = targets_acc_3.mean()
        total_target_acc_m = targets_acc_m.mean()


        if total_target_acc_1 > best_acc_1:
            best_acc_1 = total_target_acc_1
        if total_target_acc_2 > best_acc_2:
            best_acc_2 = total_target_acc_2
        if total_target_acc_3 > best_acc_3:
            best_acc_3 = total_target_acc_3
        if total_target_acc_m > best_acc_m:
            best_acc_m = total_target_acc_m
            # torch.save({'student_model': student_model.state_dict(), 'acc': best_student_acc, 'epoch': epoch},
            #            "{}/kd_da_alt_pth_student_best_model.pth".format(save_dir))
            # if save_name != "":
            #     torch.save(student_model, save_name)


        if logger is not None:
            logger.log_scalar("beta_epoch".format(logger_id), beta.item(), epoch)
            for i in range(len(targets_dloader)):
                logger.log_scalar("training_loss_t_{}".format(i), total_losses[i].item(), epoch)
            # for i in range(len(targets_dloader)):
            #     logger.log_scalar("da_loss_t_{}".format(i), da_losses[i].item(), epoch)
            # for i in range(len(targets_dloader)):
            #     logger.log_scalar("kd_loss_{}".format(i), kd_losses[i].item(), epoch)
            logger.log_scalar("da_lr_epoch".format(logger_id), new_lr_da, epoch)
            logger.log_scalar("kd_lr_epoch".format(logger_id), optimizer_kd.param_groups[0]['lr'], epoch)
            for i in range(len(targets_dloader)):
                logger.log_scalar("{}_val_target_acc_1".format(i, logger_id), targets_acc_1[i], epoch)
                logger.log_scalar("{}_val_target_acc_2".format(i, logger_id), targets_acc_2[i], epoch)
                logger.log_scalar("{}_val_target_acc_3".format(i, logger_id), targets_acc_3[i], epoch)
                logger.log_scalar("{}_val_target_acc_m".format(i, logger_id), targets_acc_m[i], epoch)
            logger.log_scalar("target_total_acc_1".format(logger_id), total_target_acc_1, epoch)
            logger.log_scalar("target_total_acc_2".format(logger_id), total_target_acc_2, epoch)
            logger.log_scalar("target_total_acc_3".format(logger_id), total_target_acc_3, epoch)
            logger.log_scalar("target_total_acc_m".format(logger_id), total_target_acc_m, epoch)

        if scheduler_da is not None:
            scheduler_da.step()

        if scheduler_kd is not None:
            scheduler_kd.step()

    return best_acc_1, best_acc_2, best_acc_3, best_acc_m

