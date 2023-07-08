from DA.mydataset import Mydataset
from torch.utils.data import DataLoader
import numpy as np

def get_source_m_target_loader(source_path, target_path_s, batch_size=64, num_classes = 3, pin_memory=False, drop_last=True):

    targets_dataloader = [0] * len(target_path_s)
    targets_testloader = [0] * len(target_path_s)

    source_dataloader = None
    if source_path != "":
        source_dataloader = loader(source_path, batch_size, num_classes, pin_memory, drop_last)
    for i, p in enumerate(target_path_s):
        targets_dataloader[i] = target_loader(p, i, batch_size, num_classes, pin_memory, drop_last)
        targets_testloader[i] = target_test_loader(p, i, batch_size, num_classes, pin_memory)

    return source_dataloader, targets_dataloader, targets_testloader

def loader(path, batch_size=64, num_classes = 3, pin_memory=False, drop_last = True):
    sample =[]
    label = []

    index = [i for i in range(15)]
    for i in index:
        t_sample = np.load(path + "person_%d data.npy" % i, allow_pickle=True)
        t_label = np.load(path + "label.npy", allow_pickle=True)
        sample.append(t_sample)
        label.append(t_label)
    sample = np.concatenate(sample, axis=0)
    label = np.concatenate(label, axis=0)
    dataset = Mydataset(sample,label,num_classes)
    data_loader =DataLoader(dataset, batch_size=batch_size, pin_memory=pin_memory, shuffle=False, drop_last = drop_last, num_workers = 0)

    return data_loader

def target_loader(path, i, batch_size=64, num_classes = 3, pin_memory=False, drop_last = True):
    sample = np.load(path + "person_%d DE.npy" % i, allow_pickle=True).transpose(0,2,3,1)
    label = np.load(path + "label.npy", allow_pickle=True)

    dataset = Mydataset(sample,label,num_classes)
    data_loader =DataLoader(dataset, batch_size=batch_size, pin_memory=pin_memory, shuffle=False, drop_last = drop_last, num_workers = 0)

    return data_loader


def target_test_loader(path, i, batch_size=64, num_classes = 3, pin_memory=False):
    sample = np.load(path + "person_%d DE.npy" % i, allow_pickle=True).transpose(0,2,3,1)
    label = np.load(path + "label.npy", allow_pickle=True)

    dataset = Mydataset(sample,label,num_classes)
    data_loader =DataLoader(dataset, batch_size=batch_size, pin_memory=pin_memory, shuffle=False, num_workers = 0)

    return data_loader




