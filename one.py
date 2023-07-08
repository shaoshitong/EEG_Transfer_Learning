import argparse
import numpy as np
from torch.utils.data import DataLoader
import DA.mydataset as mydataset
import cmodels.WideLinear as WideLinear
import cmodels.OneWideLinear as OWL
import torch.nn.functional as F
import torch
import tqdm
import time
import os


def test(model, test_dataloader):
    model.eval()
    correct_1 = 0
    correct_2 = 0
    correct_3 = 0
    correct_m = 0
    with torch.no_grad():
        for t_sample, label in tqdm.tqdm(test_dataloader):
            t_sample = t_sample.to(device)
            label = label.long()
            label = label.to(device)
            _, _, _, class_output_1, class_output_2, class_output_3, class_output_m = model(t_sample)

            t_correct_1 = torch.eq(torch.argmax(class_output_1, 1),
                                   torch.argmax(label, 1))  # get the index of the max log-probability
            t_correct_2 = torch.eq(torch.argmax(class_output_2, 1), torch.argmax(label, 1))
            t_correct_3 = torch.eq(torch.argmax(class_output_3, 1), torch.argmax(label, 1))
            t_correct_m = torch.eq(torch.argmax(class_output_m, 1), torch.argmax(label, 1))
            correct_1 += t_correct_1.sum().item()
            correct_2 += t_correct_2.sum().item()
            correct_3 += t_correct_3.sum().item()
            correct_m += t_correct_m.sum().item()

    test_acc_1 = 100. * correct_1 / len(test_dataloader.dataset)
    test_acc_2 = 100. * correct_2 / len(test_dataloader.dataset)
    test_acc_3 = 100. * correct_3 / len(test_dataloader.dataset)
    test_acc_m = 100. * correct_m / len(test_dataloader.dataset)

    return test_acc_1, test_acc_2, test_acc_3, test_acc_m


def train(model, ema_model, optimizer, train_dataloader, test_dataloader, test_person_id):
    train_result_1 = ['train_acc_1']
    train_result_2 = ['train_acc_2']
    train_result_3 = ['train_acc_3']
    train_result_m = ['train_acc_m']
    test_result_1 = ['test_acc_1']
    test_result_2 = ['test_acc_2']
    test_result_3 = ['test_acc_3']
    test_result_m = ['test_acc_m']

    best_acc_1 = -float('inf')  # ？
    best_acc_2 = -float('inf')
    best_acc_3 = -float('inf')
    best_acc_m = -float('inf')

    for epoch in range(args.epoch):
        model.train()

        correct_1 = 0
        correct_2 = 0
        correct_3 = 0
        correct_m = 0

        # tqdm: 设置进度条
        for x, label in tqdm.tqdm(train_dataloader):
            #  loss中需要long型
            label = label.long()

            # cuda
            x = x.to(device)
            _, _, _, label_pred_1, label_pred_2, label_pred_3, label_pred_m = model(x)  # 40 x 60 x 2

            with torch.no_grad():
                _, _, _, ema_label_pred_1, ema_label_pred_2, ema_label_pred_3, ema_label_pred_m = ema_model(
                    x)  # 40 x 60 x 2
            # compute loss
            label = label.to(device)
            label = label.float()
            # loss_cls_1 = F.cross_entropy(label_pred_1, label)  # TODO FLAG1
            # loss_cls_2 = F.cross_entropy(label_pred_2, label)
            # loss_cls_3 = F.cross_entropy(label_pred_3, label)l/ /
            # loss_cls_m = F.cross_entropy(label_pred_m, label)
            loss_cls_1 = F.cross_entropy(label_pred_1, label)  # TODO FLAG1
            loss_cls_2 = F.cross_entropy(label_pred_2, label)
            loss_cls_3 = F.cross_entropy(label_pred_3, label)
            loss_cls_m = F.cross_entropy(label_pred_m, label)

            def kl_div(pred_S, pred_T, temperature=1):
                pred_S = torch.log_softmax(pred_S / temperature, 1)
                with torch.no_grad():
                    pred_T = torch.softmax(pred_T / temperature, 1)
                return F.kl_div(pred_S, pred_T, reduction="batchmean") * (temperature ** 2)

            one_div = kl_div(label_pred_1, label_pred_m) + kl_div(label_pred_2, label_pred_m) + kl_div(label_pred_3,
                                                                                                       label_pred_m)
            ema_div = kl_div(label_pred_1, ema_label_pred_1) + kl_div(label_pred_2, ema_label_pred_2) + kl_div(
                label_pred_3, ema_label_pred_3)
            loss = loss_cls_1 + loss_cls_2 + loss_cls_3 + loss_cls_m + one_div * args.one_weight + ema_div * args.ema_weight
            optimizer.zero_grad()  # 梯度值清零
            loss.backward()  # 计算梯度
            optimizer.step()  # 梯度下降参数更新
            # i += 1
            for param, ema_param in zip(model.parameters(), ema_model.parameters()):
                ema_param.data.mul_(0.999).add_(param.data, alpha=0.001)

            t_correct_1 = torch.eq(torch.argmax(label_pred_1, 1),
                                   torch.argmax(label, 1))  # get the index of the max log-probability
            t_correct_2 = torch.eq(torch.argmax(label_pred_2, 1), torch.argmax(label, 1))
            t_correct_3 = torch.eq(torch.argmax(label_pred_3, 1), torch.argmax(label, 1))
            t_correct_m = torch.eq(torch.argmax(label_pred_m, 1), torch.argmax(label, 1))
            correct_1 += t_correct_1.sum().item()
            correct_2 += t_correct_2.sum().item()
            correct_3 += t_correct_3.sum().item()
            correct_m += t_correct_m.sum().item()

        # item返回tensor的元素值
        train_acc_1 = 100. * correct_1 / len(train_dataloader.dataset)
        train_acc_2 = 100. * correct_2 / len(train_dataloader.dataset)
        train_acc_3 = 100. * correct_3 / len(train_dataloader.dataset)
        train_acc_m = 100. * correct_m / len(train_dataloader.dataset)
        item_pr = 'Epoch: [{}/{}],total_loss:{:.4f},Epoch{}_Acc_1:{:.4f},Acc_2:{:.4f},Acc_3:{:.4f},Acc_4:{:.4f}' \
            .format(epoch, args.epoch, loss.item(), epoch, train_acc_1, train_acc_2, train_acc_3, train_acc_m)
        print(item_pr)

        # test
        test_acc_1, test_acc_2, test_acc_3, test_acc_m = test(model, test_dataloader)

        train_result_1.append(train_acc_1)
        train_result_2.append(train_acc_2)
        train_result_3.append(train_acc_3)
        train_result_m.append(train_acc_m)
        test_result_1.append(test_acc_1)
        test_result_2.append(test_acc_2)
        test_result_3.append(test_acc_3)
        test_result_m.append(test_acc_m)

        test_info = 'Test acc_1 Epoch_{}: {:.4f}, acc_2 :{:.4f}, acc_3 :{:.4f}, acc_m :{:.4f}'.format(epoch, test_acc_1,
                                                                                                      test_acc_2,
                                                                                                      test_acc_3,
                                                                                                      test_acc_m)
        print(test_info)

        if test_acc_1 > best_acc_1:
            best_acc_1 = test_acc_1
        if test_acc_2 > best_acc_2:
            best_acc_2 = test_acc_2
        if test_acc_3 > best_acc_3:
            best_acc_3 = test_acc_3
        if test_acc_m > best_acc_m:
            best_acc_m = test_acc_m
            # torch.save(model.state_dict(), args.model_path + "model.npy")

            # torch.save(model.state_dict(),model_name)
        elif epoch % 100 == 0 and epoch != 0:
            continue
            # torch.save(model.state_dict(), model_name)

    Best_info = 'best_acc_1: {:.4f}, best_acc_2: {:.4f}, best_acc_3: {:.4f}, best_acc_m: {:.4f}'.format(best_acc_1,
                                                                                                        best_acc_2,
                                                                                                        best_acc_3,
                                                                                                        best_acc_m)
    print(Best_info)


def main(args, device):
    data_path = "F:\\wsy\\研一\\科研\\Mymodel\\MT-MTDA - subject - one - WFE - their\\datasets\\deap_one\\"
    l_index = [i for i in range(14)]
    # l_index = l_index[1:] + [l_index[0]]
    sumT = 0
    for index, i in enumerate(l_index):

        #样本分类 ： train & test
        for j in range(14):
            content_label = np.load(data_path + "person_%d label.npy" % (j))
            content_label = np.squeeze(content_label)
            content_data = np.load(data_path + "person_%d DE.npy" % (j))
            # test
            if j == i:
                test_data_total = content_data
                test_label = content_label
            # train
            elif j == 0 or (j == 1 and i == 0):
                data_label = content_label
                data_total = content_data
            else:
                data_label = np.append(data_label, content_label, axis=0)
                data_total = np.append(data_total, content_data, axis=0)

        #dataload
        N = len(data_label)
        _index = np.random.choice(N, size=N, replace=False)
        data_total = data_total[_index]
        data_label = data_label[_index]


        train_data = mydataset.Mydataset(data_total, data_label)
        test_data = mydataset.Mydataset(test_data_total, test_label)
        train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=False, num_workers=4)
        test_dataloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True, num_workers=4)

        # 单个实验重复3次
        for trail_num in range(1):
            # 信息输出
            print_info = "=================个体{}，第{}次=======================\n".format(i, trail_num + 1)
            print(print_info)
            start_time = time.perf_counter()
            model = WideLinear.WLE(
                num_classes=args.num_classes,
                channels=32,
                hidden_dim=args.hidden_dim,
                d_model=args.d_model,
                grad_rate=args.grad_rate,
                use_inf=True
            )
            model = model.to(device)
            ema_model = WideLinear.WLE(
                num_classes=args.num_classes,
                channels=32,
                hidden_dim=args.hidden_dim,
                d_model=args.d_model,
                grad_rate=args.grad_rate,
                use_inf=True
            )
            ema_model = ema_model.to(device)
            ema_model.load_state_dict(model.state_dict())

            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)  # TODO: Modified
            train(model, ema_model, optimizer, train_dataloader, test_dataloader, i)
            end_time = time.perf_counter()
            time_sum = end_time - start_time
            sumT = sumT + time_sum
            test_info = 'Subject {} time: {} l SUMTIME : {}\n'.format(i, time_sum, sumT)

            save_dir = "models_one_their"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            file_path = os.path.join(save_dir, f"one_model_14_{i}.pth")
            torch.save(model.state_dict(), file_path)  # save: state_dict() load: load_state_dict()
            print(test_info)


if __name__ == "__main__":
    device = torch.device('cuda')

    parser = argparse.ArgumentParser()  # 创建对象
    parser.add_argument('--cuda', type=str, default='0')
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--d_model', type=int, default=100)
    parser.add_argument('--grad_rate', type=float, default=0.9)
    parser.add_argument('--one-weight', type=float, default=0.5)
    parser.add_argument('--ema-weight', type=float, default=2)
    args = parser.parse_args()

    main(args, device)
