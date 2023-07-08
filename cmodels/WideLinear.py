import torch.nn as nn
import torch
import torch.nn.functional as F
import math
import einops

device = torch.device("cuda")
class INF(nn.Module):
    def __init__(self, in_channels, hidden_dim, d_model):
        super(INF, self).__init__()
        self.in_channels = in_channels
        # 上采样操作，提高分辨率
        self.interploate = lambda x: F.interpolate(x, [d_model, d_model], mode="bilinear")  # 利用双线性函数进行采样
        self.relu = nn.ReLU(inplace=False)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.linear = nn.Linear(in_channels * d_model, hidden_dim)

    def forward(self, x):
        # bs,ic,bn,to -> bs,ic,to,to
        x = self.interploate(x)
        x = einops.rearrange(x, "b i h w -> b w (i h)")
        x = self.linear(x)
        x = einops.rearrange(x, "b i c -> b c i")
        return x


class WLE(nn.Module):
    def __init__(self, num_classes, channels,
                 hidden_dim, d_model, grad_rate=1.0, use_inf=False):
        super(WLE, self).__init__()
        self.classes = num_classes
        self.use_inf = use_inf
        self.channels = channels
        self.hidden_dim = hidden_dim
        self.d_model = d_model
        self.inf = INF(self.channels, self.hidden_dim, self.d_model)
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())
        self.grad_rate = grad_rate
        self.downsamples = nn.ModuleList([nn.Conv1d(self.hidden_dim, self.hidden_dim * 2, kernel_size=4, stride=4, padding=0,
                                    groups=self.hidden_dim) for i in range(3)])
        self.conv = nn.Linear(hidden_dim * d_model, 3)
        self.bn_v1 = nn.BatchNorm1d(hidden_dim * d_model)
        self.fcs = nn.ModuleList([nn.Linear(hidden_dim * d_model // 2, num_classes) for i in range(3)])
        self.relus = nn.ModuleList([nn.ReLU(inplace=False) for i in range(3)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for i in range(3)])

    def forward(self, data):
        data = data.unsqueeze(3)
        data = self.inf(data).float()
        data_trans = einops.rearrange(data, "b h d -> b (h d)")

        g_c = F.relu(data_trans)
        g_c = self.bn_v1(g_c)
        g_c = self.conv(g_c)
        g_c = F.softmax(g_c, dim=1)

        data_1 = self.bns[0](self.relus[0](data))
        data_2 = self.bns[1](self.relus[1](data))
        data_3 = self.bns[2](self.relus[2](data))
        data_1 = self.downsamples[0](data_1) / 4
        data_2 = self.downsamples[1](data_2) / 4
        data_3 = self.downsamples[2](data_3) / 4
        data_1 = data_1.view(data_1.shape[0], -1)
        data_2 = data_2.view(data_2.shape[0], -1)
        data_3 = data_3.view(data_3.shape[0], -1)
        data_1_fc = self.fcs[0](data_1)
        data_2_fc = self.fcs[1](data_2)
        data_3_fc = self.fcs[2](data_3)
        g_c_1 = g_c[:, 0][..., None]
        g_c_2 = g_c[:, 1][..., None]
        g_c_3 = g_c[:, 2][..., None]
        data_m_fc = g_c_1 * data_1_fc + g_c_2 * data_2_fc + g_c_3 * data_3_fc

        return data_1, data_2, data_3, data_1_fc, data_2_fc, data_3_fc, data_m_fc


def WLE_extractor():
    model = WLE(3, 32, 64, 100, 0.9, True)
    return model

# class WLE_DEAP_A(nn.Module):
#     def __init__(self, num_classes, out, band_nums, time_dim,
#                  hidden_dim, d_model, grad_rate=1.0, use_inf=False):
#         super(WLE_DEAP_A, self).__init__()
#         self.classes = num_classes
#         self.use_inf = use_inf
#         self.time_out = time_dim
#         self.band_nums = band_nums
#         self.in_channels = out // band_nums
#         self.hidden_dim = hidden_dim
#         self.d_model = d_model
#         self.inf = INF(self.in_channels, self.band_nums, self.time_out, self.hidden_dim, self.d_model)
#         self.blocks = []
#         self.blocks.append(nn.Identity())
#         self.blocks = nn.ModuleList(self.blocks)
#         # self.pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())
#         self.grad_rate = grad_rate
#         self.downsample = nn.Conv2d(self.hidden_dim, self.hidden_dim,
#                                     kernel_size=(4, 4), stride=(4, 4),
#                                     padding=(0, 0),
#                                     groups=self.hidden_dim,
#                                     )
#
#     def forward(self, data):
#         data = self.inf(data).float()
#         for block in self.blocks:
#             data = block(data)
#         data = self.downsample(data)/4
#         m = data.shape[-1]
#         data = data.view(data.shape[0], -1)/(m*self.grad_rate)
#         return data
#
#
# class WLE_DEAP_V(nn.Module):
#     def __init__(self, num_classes, out, band_nums, time_dim,
#                  hidden_dim, d_model, grad_rate=1.0, use_inf=False):
#         super(WLE_DEAP_V, self).__init__()
#         self.classes = num_classes
#         self.use_inf = use_inf
#         self.time_out = time_dim
#         self.band_nums = band_nums
#         self.in_channels = out // band_nums
#         self.hidden_dim = hidden_dim
#         self.d_model = d_model
#         self.inf = INF(self.in_channels, self.band_nums, self.time_out, self.hidden_dim, self.d_model)
#         self.blocks = []
#         self.blocks.append(nn.Identity())
#         self.blocks = nn.ModuleList(self.blocks)
#         # self.pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())
#         self.grad_rate = grad_rate
#         # self.downsample这个可以调试
#         self.downsample = nn.Conv2d(self.hidden_dim, self.hidden_dim,
#                                     kernel_size=(4, 4), stride=(4, 4),
#                                     padding=(0, 0),
#                                     groups=self.hidden_dim
#                                     )
#
#     def forward(self, data):
#         data = self.inf(data).float()
#         for block in self.blocks:
#             data = block(data)
#         data = self.downsample(data)/4
#         m = data.shape[-1]
#         data = data.view(data.shape[0], -1)/(m*self.grad_rate)
#         return data
