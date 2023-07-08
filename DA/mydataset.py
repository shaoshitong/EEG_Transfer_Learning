from torch.utils.data import Dataset
import torch
import numpy as np

class Mydataset(Dataset):
    def __init__(self, dataset, label, num_classes = 3):
        super(Dataset, self).__init__()
        self.dataset = dataset
        self.label = label.astype(np.longlong)
        self.num_classes = num_classes

    def __getitem__(self, index):
        sample = self.dataset[index,:,:]
        sample = torch.Tensor(sample)
        label = torch.zeros(self.num_classes)
        label[self.label[index]] = 1
        return sample, label

    def __len__(self):
        return len(self.label)

class Datasampler(Dataset):
    def __init__(self, data ,label):
        self.data = torch.Tensor(data)
        self.label = label.astype(np.longlong)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index,:,:]
        label = self.label[index]
        return data,label