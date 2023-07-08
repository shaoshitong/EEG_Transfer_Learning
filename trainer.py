import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import sys
from torchvision.models import resnet
from utils.utils import Logger, save_checkpoint, AverageMeter, accuracy, PR_score
from dataset.dataset import DimpleDataset
import torch.nn.functional as F
from torch.autograd import Variable

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'      # 使用 GPU 3


models = {'resnet18':512, 'resnet34':512, 'resnet50':2048, 'resnet101':2048}
model_names = list(models.keys())
parser = argparse.ArgumentParser(description='Propert ResNets for mini-nico in pytorch')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet34',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) +
                    ' (default: resnet34)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=1, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=1, type=int,
                    metavar='N', help='print frequency (default: 50)')
parser.add_argument('--test-print-freq', '-tp', default=81, type=int,
                    metavar='N', help='print frequency (default: 50)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--half', dest='half', action='store_true',
                    help='use half-precision(16-bit) ')
parser.add_argument('--gpu', dest='gpu', action='store_true',
                    help='whether choose the gpu to train')
parser.add_argument('--save-dir', dest='save_dir',
                    help='The directory used to save the trained models',
                    default='save_temp', type=str)
parser.add_argument('--val-every', dest='val_every',
                    help='val the checkpoint at every specified number of epochs',
                    type=int, default=5)
parser.add_argument('--save-every', dest='save_every',
                    help='Saves checkpoints at every specified number of epochs',
                    type=int, default=20)
parser.add_argument('--train_data_txt_path', type=str, default='train.txt')
parser.add_argument('--val_data_txt_path', type=str, default='val.txt')
parser.add_argument('--p_percent', default=0.7, type=float, help='precison percent')
parser.add_argument('--r_percent', default=0.3, type=float, help='recall percent')
  
best_score = 0


class LDAMLoss(nn.Module):
    
    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30):
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list)) # nj的四次开方
        m_list = m_list * (max_m / np.max(m_list)) # 常系数 C
        m_list = torch.cuda.FloatTensor(m_list) # 转成 tensor
        self.m_list = m_list
        assert s > 0
        self.s = s # 这个参数的作用论文里提过么？
        self.weight = weight # 和频率相关的 re-weight

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8) # 和 x 维度一致全 0 的tensor
        index.scatter_(1, target.data.view(-1, 1), 1) # dim idx input
        index_float = index.type(torch.cuda.FloatTensor) # 转 tensor
        ''' 以上的idx指示的应该是一个batch的y_true '''
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0,1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m # y 的 logit 减去 margin
        output = torch.where(index, x_m, x) # 按照修改位置合并
        return F.cross_entropy(self.s*output, target, weight=self.weight)


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()


def main():
    global args, best_score
    args = parser.parse_args()
    record_time = time.localtime(time.time())
    if not os.path.exists("logs/"):
        os.makedirs("logs/")
    
    # Check the save_dir exists or not
    args.save_dir = "models/save_" + args.arch + '_{}'.format(time.strftime('%m-%d-%H-%M', record_time)) + "/"

    # model = resnet().__dict__[args.arch]()
    model = resnet.resnet34(pretrained=True)
    model.fc = nn.Linear(in_features=models[args.arch], out_features=2)
    if args.gpu:
        model.cuda()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_score']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.evaluate, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    
    
    labels_dict = {
        '非Dimple': 0,
        'Dimple': 1
    }     

    train_dataset = DimpleDataset(args.train_data_txt_path, labels_dict)
    val_dataset = DimpleDataset(args.val_data_txt_path, labels_dict, train=False)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # define loss function (criterion) and optimizer
    if args.gpu:
        # criterion = nn.CrossEntropyLoss().cuda()
        # criterion = FocalLoss().cuda()
        criterion = LDAMLoss().cuda()
    else:
        # criterion = nn.CrossEntropyLoss()
        # criterion = FocalLoss()
        criterion = LDAMLoss()


    if args.half:
        model.half()
        criterion.half()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=[50, 75])

    if args.arch in ['resnet1202', 'resnet110']:
        # for resnet1202 original paper uses lr=0.01 for first 400 minibatches for warm-up
        # then switch back. In this setup it will correspond for first epoch.
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr*0.1


    val_score = []
    for epoch in range(args.start_epoch, args.epochs + 1):

        # train for one epoch
        print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
        train(args, train_loader, model, criterion, optimizer, epoch)
        lr_scheduler.step()

        # evaluate on validation set
        if epoch > 0 and epoch % args.val_every == 0:
            score = validate(args, val_loader, model, criterion)
            val_score.append(score)
            # remember best prec@1 and save checkpoint
            is_best = score > best_score
            best_score = max(score, best_score)

        if epoch > 0 and epoch % args.save_every == 0:
            sys.stdout = Logger("logs/" + args.arch + "_train_info_{}.txt".format(time.strftime('%m-%d-%H-%M', record_time)))
            if not os.path.exists(args.save_dir):
                os.makedirs(args.save_dir)

            save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'best_score': best_score,
            }, is_best, filename=os.path.join(args.save_dir, 'checkpoint_'+ str(epoch) +'.pt'))

    print("best_score_{}".format(max(val_score)))


def train(args, train_loader, model, criterion, optimizer, epoch):
    """
        Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu:
            target_var = target.cuda()
            input_var = input.cuda()

        else:
            target_var = target
            input_var = input

        if args.half:
            input_var = input_var.half()

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target_var)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (i+1) % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, (i+1), len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1))


def validate(args, val_loader, model, criterion):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    output_list = list()
    gt_list = list()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            if args.gpu:
                input_var = input.cuda()
                target_var = target.cuda()
            else:
                input_var = input
                target_var = target

            if args.half:
                input_var = input_var.half()

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            output = output.float()
            loss = loss.float()

            # record loss
            losses.update(loss.item(), input.size(0))

            # measure PR
            _, pred = output.topk(1, 1, True, True)
            pred = pred[0][0].cpu().numpy()
            gt = target_var[0].cpu().numpy()
            output_list.append(pred)
            gt_list.append(gt)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if (i+1) % args.test_print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                          (i+1), len(val_loader), batch_time=batch_time, loss=losses))

    precision, recall, score = PR_score(output_list, gt_list, args.p_percent, args.r_percent)
    print(' *precision {precision:.3f}\t*recall {recall:.3f}\t*score {score:.3f}'
          .format(precision=precision, recall=recall, score=score))

    return score



if __name__ == '__main__':
     main()
