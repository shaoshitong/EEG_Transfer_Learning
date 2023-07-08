import torch
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F
from DA.GRL_utils import ReverseLayerGrad

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, delta):
        ctx.delta = delta
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.delta
        return output, None

class Discriminator(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=256):
        super(Discriminator, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dis1 = nn.Linear(input_dim, hidden_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.dis2 = nn.Linear(hidden_dim, 1)


    def forward(self, x):
        x = F.relu(self.dis1(x))
        x = self.dis2(self.bn(x))
        x = torch.sigmoid(x)
        return x

class DANN_GRL_Resnet(nn.Module):
    def __init__(self, resnet_func, num_classes):
        super(DANN_GRL_Resnet, self).__init__()

        self.sharedNet = resnet_func()
        self.domain_classifier = Discriminator(input_dim=self.sharedNet.fc.in_features, hidden_dim=4096)
        #self.domain_classifier_ensemble = Discriminator(input_dim=self.sharedNet.fc.out_features, hidden_dim=1024)
        self.cls_fc = nn.Linear(self.sharedNet.fc.in_features, num_classes)

    def forward(self, input, delta=1, source=True):
        (input_1, input_2, input_3, clc_1, clc_2, clc_3, input_m) = self.sharedNet(input)
        features_1 = input_1.view(input.size(0), -1)
        features_2 = input_2.view(input.size(0), -1)
        features_3 = input_3.view(input.size(0), -1)
        features_m = input_m.view(input.size(0), -1)

        class_output_1 = clc_1
        class_output_2 = clc_2
        class_output_3 = clc_3
        class_output_m = features_m
        loss_adv_1 = self.get_adversarial_result(
            features_1, source, delta)
        loss_adv_2 = self.get_adversarial_result(
            features_2, source, delta)
        loss_adv_3 = self.get_adversarial_result(
            features_3, source, delta)
        #loss_adv_m = self.get_adversarial_result_ensemble(features_m, source, delta)

        return class_output_1, class_output_2, class_output_3, class_output_m, loss_adv_1, loss_adv_2, loss_adv_3

    def get_adversarial_result(self, x, source=True, delta=1):
        loss_fn = nn.BCELoss()
        if source:
            domain_label = torch.ones(len(x)).long().to(x.device).unsqueeze(1)
        else:
            domain_label = torch.zeros(len(x)).long().to(x.device).unsqueeze(1)
        x = ReverseLayerF.apply(x, delta)
        domain_pred = self.domain_classifier(x)
        loss_adv = loss_fn(domain_pred, domain_label.float())
        return loss_adv

    def get_adversarial_result_ensemble(self, x, source=True, delta=1):
        loss_fn = nn.BCELoss()
        if source:
            domain_label = torch.ones(len(x)).long().to(x.device).unsqueeze(1)
        else:
            domain_label = torch.zeros(len(x)).long().to(x.device).unsqueeze(1)
        x = ReverseLayerF.apply(x, delta)
        domain_pred = self.domain_classifier_ensemble(x)
        loss_adv = loss_fn(domain_pred, domain_label.float())
        return loss_adv

    def nforward(self, source):
        _, _, _, source_1, source_2, source_3, source_m = self.sharedNet(source)
        source_1 = source_1.view(source_1.size(0), -1)
        source_2 = source_2.view(source_2.size(0), -1)
        source_3 = source_3.view(source_3.size(0), -1)
        source_m = source_m.view(source_m.size(0), -1)
        return source_1, source_2, source_3, source_m

class DANN_GRL_WLE(nn.Module):
    def __init__(self, WLE, num_classes):
        super(DANN_GRL_WLE, self).__init__()

        self.sharedNet = WLE()
        self.domain_classifier = Discriminator(input_dim=self.sharedNet.fcs[0].in_features, hidden_dim=1024)
        # self.cls_fc = nn.Linear(self.sharedNet.fc.in_features, num_classes)

    def forward(self, input, delta=1, source=True):
        (input_1, input_2, input_3, clc_1, clc_2, clc_3, input_m) = self.sharedNet(input)
        features_1 = input_1.view(input.size(0), -1)
        features_2 = input_2.view(input.size(0), -1)
        features_3 = input_3.view(input.size(0), -1)

        loss_adv_1 = self.get_adversarial_result(
            features_1, source, delta)
        loss_adv_2 = self.get_adversarial_result(
            features_2, source, delta)
        loss_adv_3 = self.get_adversarial_result(
            features_3, source, delta)

        return clc_1, clc_2, clc_3, input_m, loss_adv_1, loss_adv_2, loss_adv_3

    def get_adversarial_result(self, x, source=True, delta=1):
        loss_fn = nn.BCELoss()
        if source:
            domain_label = torch.ones(len(x)).long().to(x.device).unsqueeze(1)
        else:
            domain_label = torch.zeros(len(x)).long().to(x.device).unsqueeze(1)
        x = ReverseLayerF.apply(x, delta)
        domain_pred = self.domain_classifier(x)
        loss_adv = loss_fn(domain_pred, domain_label.float())
        return loss_adv

    def nforward(self, source):
        _, _, _, source_1, source_2, source_3, source_m = self.sharedNet(source)
        return source_1, source_2, source_3, source_m



