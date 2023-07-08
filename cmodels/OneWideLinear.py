import torch.nn as nn
import torch
import torch.nn.functional as F
import math
import einops


class INF(nn.Module):
    def __init__(self, in_channels, hidden_dim, d_model):
        super(INF, self).__init__()
        self.in_channels = in_channels
        # 上采样操作，提高分辨率
        self.interploate = lambda x: F.interpolate(x, [d_model, d_model], mode="bilinear")
        self.relu = nn.ReLU(inplace=False)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.linear = nn.Linear(in_channels * d_model, hidden_dim)

    def forward(self, x):
        # bs,ic,bn,to -> bs,ic,to,to
        x = self.interploate(x)
        x = einops.rearrange(x, "b i h w -> b w (i h)")
        x = self.linear(x)
        x = einops.rearrange(x, "b i c -> b c i")
        x = self.bn(self.relu(x))
        x = x.unsqueeze(-1)
        xs = x.repeat(1, 1, 1, x.shape[2])
        return xs


class WLE(nn.Module):
    def __init__(self, num_classes, channels, use_inf,
                 hidden_dim, d_model, grad_rate=1.0):
        super(WLE, self).__init__()
        self.classes = num_classes
        self.channels = channels
        self.hidden_dim = hidden_dim
        self.d_model = d_model
        if use_inf:
            self.inf = INF(self.channels, self.hidden_dim, self.d_model)
        else:
            self.inf = nn.Identity()
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())
        self.grad_rate = grad_rate
        self.downsample = nn.Conv2d(self.hidden_dim, self.hidden_dim * 2, kernel_size=(4, 4), stride=(4, 4),
                                    padding=(0, 0), groups=self.hidden_dim)
        self.fc = nn.Linear(self.hidden_dim * 2 * d_model * d_model // 16, 3)
        self.classifiers = nn.ModuleList([nn.Linear(self.hidden_dim * 2 * d_model * d_model // 16, num_classes) for i in range(3)])

    def forward(self, data):
        data = self.inf(data).float()
        data = self.downsample(data) / 4
        m = data.shape[-1]
        data = data.view(data.shape[0], -1) / (m * self.grad_rate)
        g_c = F.softmax(self.fc(data), dim=1)
        output = []
        for classifier in self.classifiers:
            output += [classifier(data)]
        data_m_fc = g_c[..., 0][..., None] * output[0] + g_c[..., 1][..., None] * output[1] + g_c[..., 2][..., None] * \
                    output[2]
        return data, data, data, output[0], output[1], output[2], data_m_fc
