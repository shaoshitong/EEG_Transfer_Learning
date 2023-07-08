import os
from utils import get_config_var
vars = get_config_var()

from sacred import Experiment
ex = Experiment()

import ONE_kd_da_grl_alt_multi_target_cst_fac

import torch
import cmodels.ResNet as ResNet
import cmodels.WideLinear as WLE
import cmodels.alexnet as AlexNet
import cmodels.LeNet as Lenet
import cmodels.DAN_model as DAN_model
import DA.DA_datasets_deap_sead as DA_datasets
import torch.nn as nn
import torch.nn.functional as F
import cmodels.ONE_DANN_GRL as ONE_DANN_GRL
from utils import LoggerForSacred

@ex.config
def exp_config():

    #Hyper Parameters Config
    init_lr_da =  0.001
    init_lr_kd = 0.001
    momentum = 0.9
    weight_decay = 5e-4
    device = "cuda"
    epochs = 50
    batch_size = 16
    init_beta = 0.1
    end_beta = 0.9
    T = 20
    alpha = 0.2
    gamma = 0.5
    batch_norm = True
    is_cst = True
    resize_digits = 28

    #Scheduler
    is_scheduler_da = True
    is_scheduler_kd = True
    scheduler_kd_fn = torch.optim.lr_scheduler.MultiStepLR
    scheduler_kd_steps = [250, 350]
    scheduler_kd_gamma = 0.1

    #Dataset config
    dataset_name = ""
    source_dataset_path = ""
    target_dataset_paths = []

    #Model Config
    dan_model_func = ONE_DANN_GRL.DANN_GRL_WLE
    model_net_func = WLE.WLE_extractor()

    #Debug config
    is_debug = False

@ex.capture()
def exp_kd_da_grl_alt(init_lr_da, init_lr_kd, momentum, weight_decay, device, epochs, batch_size, init_beta, end_beta, T, alpha, gamma, batch_norm, is_cst,
                      is_scheduler_da, is_scheduler_kd, scheduler_kd_fn, scheduler_kd_steps, scheduler_kd_gamma,
                      source_dataset_path, target_dataset_paths, num_classes,
                      dan_model_func, model_net_func,
                      is_debug, _run):

    source_dataloader, targets_dataloader, targets_testloader = DA_datasets.get_source_m_target_loader(
                                                                                                   source_dataset_path,
                                                                                                   target_dataset_paths,
                                                                                                   batch_size,
                                                                                                   num_classes,0,drop_last=True)

    model = dan_model_func(model_net_func, source_dataloader.dataset.num_classes).to(device)


    logger = LoggerForSacred(None, ex)

    growth_rate = torch.zeros(1)
    if init_beta != 0.0:
        growth_rate = torch.log(torch.FloatTensor([end_beta / init_beta])) / torch.FloatTensor([epochs])

    optimizer_da = torch.optim.SGD(model.parameters(), init_lr_da,
                                momentum=momentum, weight_decay=weight_decay)

    optimizer_kd = torch.optim.SGD(model.parameters(), init_lr_kd,
                            momentum=momentum, weight_decay=weight_decay)

    if scheduler_kd_fn is not None:
        scheduler_kd = scheduler_kd_fn(optimizer_kd, scheduler_kd_steps, scheduler_kd_gamma)


    best_acc_1, best_acc_2, \
    best_acc_3, best_acc_m = ONE_kd_da_grl_alt_multi_target_cst_fac.grl_multi_target_hinton_alt(init_lr_da, init_lr_kd, device, epochs, T, alpha, gamma, growth_rate, init_beta,
                                                source_dataloader, targets_dataloader, targets_testloader, optimizer_da, optimizer_kd,
                                                model,
                                                logger=logger,
                                                is_scheduler_da=is_scheduler_da, is_scheduler_kd=is_scheduler_kd, scheduler_kd=None, scheduler_da=None,
                                                is_debug=is_debug, run=_run, batch_norm=batch_norm, is_cst=is_cst)


    return best_acc_1, best_acc_2, best_acc_3, best_acc_m

@ex.main
def run_exp():
    return exp_kd_da_grl_alt()

if __name__ == "__main__":

    if os.name == 'nt':
        deap_0 = "F:\\wsy\\研一\\科研\\Mymodel\\MT-MTDA - subject - one - WFE - their\\datasets\\deap_3class_14\\subject2-0\\"
        deap_1 = "F:\\wsy\\研一\\科研\\Mymodel\\MT-MTDA - subject - one - WFE - their\\datasets\\deap_3class_14\\subject5-1\\"
        deap_2 = "F:\\wsy\\研一\\科研\\Mymodel\\MT-MTDA - subject - one - WFE - their\\datasets\\deap_3class_14\\subject10-2\\"
        deap_3 = "F:\\wsy\\研一\\科研\\Mymodel\\MT-MTDA - subject - one - WFE - their\\datasets\\deap_3class_14\\subject11-3\\"
        deap_4 = "F:\\wsy\\研一\\科研\\Mymodel\\MT-MTDA - subject - one - WFE - their\\datasets\\deap_3class_14\\subject12-4\\"
        deap_5 = "F:\\wsy\\研一\\科研\\Mymodel\\MT-MTDA - subject - one - WFE - their\\datasets\\deap_3class_14\\subject13-5\\"
        deap_6 = "F:\\wsy\\研一\\科研\\Mymodel\\MT-MTDA - subject - one - WFE - their\\datasets\\deap_3class_14\\subject14-6\\"
        deap_7 = "F:\\wsy\\研一\\科研\\Mymodel\\MT-MTDA - subject - one - WFE - their\\datasets\\deap_3class_14\\subject15-7\\"
        deap_8 = "F:\\wsy\\研一\\科研\\Mymodel\\MT-MTDA - subject - one - WFE - their\\datasets\\deap_3class_14\\subject19-8\\"
        deap_9 = "F:\\wsy\\研一\\科研\\Mymodel\\MT-MTDA - subject - one - WFE - their\\datasets\\deap_3class_14\\subject22-9\\"
        deap_10 = "F:\\wsy\\研一\\科研\\Mymodel\\MT-MTDA - subject - one - WFE - their\\datasets\\deap_3class_14\\subject24-10\\"
        deap_11 = "F:\\wsy\\研一\\科研\\Mymodel\\MT-MTDA - subject - one - WFE - their\\datasets\\deap_3class_14\\subject26-11\\"
        deap_12 = "F:\\wsy\\研一\\科研\\Mymodel\\MT-MTDA - subject - one - WFE - their\\datasets\\deap_3class_14\\subject28-12\\"
        deap_13 = "F:\\wsy\\研一\\科研\\Mymodel\\MT-MTDA - subject - one - WFE - their\\datasets\\deap_3class_14\\subject31-13\\"
        seed_session_1 = "F:\\wsy\\研一\\科研\\Mymodel\\MT-MTDA - subject - one - WFE - their\\datasets\\seed_185_32cha\\session_1\\"



    ex.run(config_updates={
                           'source_dataset_path': seed_session_1,
                           'target_dataset_paths': [deap_0, deap_1, deap_2, deap_3, deap_4,
                                                    deap_5, deap_6, deap_7, deap_8, deap_9,
                                                    deap_10, deap_11, deap_12, deap_13],
                           'init_beta': 0.1,
                           'end_beta': 0.5,
                           'init_lr_da': 0.001,
                           'init_lr_kd': 0.01,
                           'T': 3,
                           'alpha': 0.5,
                           'epochs': 100,
                           'batch_size': 128,
                           'num_classes': 3,
                           'model_net_func': WLE.WLE_extractor,
                           'dan_model_func': ONE_DANN_GRL.DANN_GRL_WLE,
                           },
           options={"--name": 'Search_cst_Pr_2_Ar_Rw_Cl_rerun_kd_da_alt_resnet50-alexnet'})
