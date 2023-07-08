import DA.mydataset as mydataset
from torch.utils.data import DataLoader
import numpy as np
from DA.Sampler import RandomIdentitySampler

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
    samples = []
    labels = []

    index = [i for i in range(15)]
    for i in index:
        t_sample = np.load(path + "person_%d DE.npy" % i, allow_pickle=True).transpose(1,0,2)
        t_label = np.load(path + "label.npy", allow_pickle=True)
        samples.append(t_sample)
        labels.append(t_label)
    samples = np.concatenate(samples, axis=0)
    labels = np.concatenate(labels, axis=0)

    N = len(labels)
    source_set_index = np.random.choice(N, size=N, replace=True)
    samples = samples[source_set_index]
    labels = labels[source_set_index]

    dataset = mydataset.Mydataset(samples, labels, num_classes)
    data_loader = DataLoader(dataset, batch_size=batch_size, pin_memory=pin_memory, shuffle=False, drop_last=drop_last,num_workers=0)
    return data_loader

def target_loader(path, i, batch_size=64, num_classes = 3, pin_memory=False, drop_last = True):
    sample = np.load(path + "person_%d DE.npy" % i, allow_pickle=True)
    label = np.load(path + "person_%d label.npy" % i, allow_pickle=True)
    label = np.squeeze(label)

    N = len(label)
    source_set_index = np.random.choice(N, size=N, replace=True)
    sample = sample[source_set_index]
    label = label[source_set_index]

    dataset = mydataset.Mydataset(sample,label,num_classes)
    data_loader =DataLoader(dataset, batch_size=batch_size, pin_memory=pin_memory, shuffle=False, drop_last = drop_last, num_workers = 0)

    return data_loader

def target_test_loader(path, i, batch_size=64, num_classes = 3, pin_memory=False):
    sample = np.load(path + "person_%d DE.npy" % i, allow_pickle=True)
    label = np.load(path + "person_%d label.npy" % i, allow_pickle=True)
    label = np.squeeze(label)

    N = len(label)
    source_set_index = np.random.choice(N, size=N, replace=True)
    sample = sample[source_set_index]
    label = label[source_set_index]

    dataset = mydataset.Mydataset(sample,label,num_classes)
    data_loader =DataLoader(dataset, batch_size=batch_size, pin_memory=pin_memory, shuffle=False, num_workers = 0)

    return data_loader




