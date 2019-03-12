import argparse
import os, sys
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models
from utils import convert_secs2time, time_string, time_file_str
from ori_model.resnet_cifar10 import ResNet18,ResNet34,ResNet50,ResNet101,ResNet152
from datetime import datetime

#from models import print_log
import ori_model
import random
import numpy as np

# model_names = sorted(name for name in models.__dict__
#     if name.islower() and not name.startswith("__")
#     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', metavar='DIR',default='./',
                    help='path to dataset')
parser.add_argument('--save_dir', type=str, default='./record/aux_resnet', help='Folder to save checkpoints and log.')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet',
                    help='model architecture: ' +
                        ' | ' +
                        ' (default: resnet18)')
parser.add_argument('--prefix', type=str, default='18', help='')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=100, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int, metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=200, type=int, metavar='N', help='print frequency (default: 100)')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('--n_gpus', type=int, default=1, help='GPU numbers')

args = parser.parse_args()
args.use_cuda = torch.cuda.is_available()

args.prefix = time_file_str()

def main():
    best_prec1 = 0

    if not os.path.isdir(args.save_dir):
      os.makedirs(args.save_dir)
    log = open(os.path.join(args.save_dir, '{}.{}.log'.format(args.arch,args.prefix)), 'w')

    # version information
    print_log("PyThon  version : {}".format(sys.version.replace('\n', ' ')), log)
    # print_log("PyTorch version : {}".format(torch.__version__), log)
    print_log("cuDNN   version : {}".format(torch.backends.cudnn.version()), log)
    # print_log("Vision  version : {}".format(torchvision.__version__), log)
    # create model
    print_log("=> creating model '{}'".format(args.arch), log)
    # model = models.__dict__[args.arch](pretrained=False)
    # print_log("=> Model : {}".format(model), log)
    print_log("=> parameter : {}".format(args), log)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    model = ResNet50(lr = args.lr,momentum=args.momentum, weight_decay=args.weight_decay,criterion = criterion,n_gpus=args.n_gpus)

    if args.use_cuda:
        model = model.cuda()

    cudnn.benchmark = True

    train_loader, test_loader = get_cifar10()

    s_time = datetime.now()
    start_time = time.time()
    epoch_time = AverageMeter()
    index = 0
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(model, epoch)

        need_hour, need_mins, need_secs = convert_secs2time(epoch_time.val * (args.epochs-epoch))
        need_time = '[Need: {:02d}:{:02d}:{:02d}]'.format(need_hour, need_mins, need_secs)
        print_log(' [{:s}] :: {:3d}/{:3d} ----- [{:s}] {:s}'.format(args.arch, epoch, args.epochs, time_string(), need_time), log)

        index = index % (len(model.seg_optimizer)-1)
        # train for one epoch
        train(train_loader, model, epoch, log,index)
        # evaluate on validation set
        val_acc_2 = validate(test_loader, model, criterion, log)

        # remember best prec@1 and save checkpoint
        is_best = val_acc_2 > best_prec1
        best_prec1 = max(val_acc_2, best_prec1)
        save_checkpoint(model=model,
                        layers=model.layers,
                        aux=model.aux,
                        seg_optimizer=model.seg_optimizer,
                        aux_optimizer=model.aux_optimizer,
                        epoch=epoch,
                        is_best= is_best)
        # print("epoch:{} accuracy: {:.4f}".format(epoch, val_acc_2))
        print_log("~~~~~best accuracy: {:.4f}".format(best_prec1),log)

        # measure elapsed time
        epoch_time.update(time.time() - start_time)
        start_time = time.time()
        index = index+1
    e_time = datetime.now()
    print_log("All time is: {}".format(e_time-s_time),log)
    log.close()

# train function (forward, backward, update)
def train(train_loader, model, epoch, log,index):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    # switch to train model

    for i in range(len(model.seg_optimizer)):
        model.layers[i].train()
        if i != index and i != len(model.seg_optimizer) - 1:
            model.aux[i].eval()
        else:
            model.aux[i].train()
        # model.aux[i].train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.use_cuda:
            target = target.cuda(async=True)
            input = input.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        # forward
        outputs, losses = model.forward2(input_var, target_var)
        # backward
        model.backward2(losses,index)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs[-1].data, target, topk=(1, 5))
        loss.update(losses[-1].item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print_log('  Epoch: [{:03d}][{:03d}/{:03d}]   '
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})   '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f})   '
                        'Loss {loss.val:.4f} ({loss.avg:.4f})   '
                        'Prec@1 {top1.val:.3f} ({top1.avg:.3f})   '
                        'Prec@5 {top5.val:.3f} ({top5.avg:.3f})   '.format(
                        epoch, i, len(train_loader), batch_time=batch_time,
                        data_time=data_time, loss=loss, top1=top1, top5=top5) + time_string(), log)
    print_log('  **Train** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'.format(top1=top1, top5=top5, error1=100-top1.avg), log)
    return top1.avg, loss.avg

def validate(val_loader, model, criterion, log):
    loss = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    for i in range(len(model.seg_optimizer)):
        model.layers[i].eval()
        model.aux[i].eval()

    for i, (input, target) in enumerate(val_loader):
        if args.use_cuda:
            target = target.cuda(async=True)
            input = input.cuda()
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        outputs, losses = model.forward2(input_var, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs[-1].data, target, topk=(1, 5))
        loss.update(losses[-1].data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

    print_log('  **Test** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'.format(top1=top1, top5=top5, error1=100-top1.avg), log)

    return top1.avg

def get_cifar10():
    if torch.cuda.is_available():
        use_cuda = True
    else:
        use_cuda = False
    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}

    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.RandomCrop(32),
                           transforms.RandomHorizontalFlip(),
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                       ])),
        batch_size=args.batch_size, shuffle=False, **kwargs)

    return train_loader, test_loader


def list2sequential(model):
    if isinstance(model, list):
        model = nn.Sequential(*model)
    return model

def save_checkpoint(model,layers,aux,seg_optimizer,aux_optimizer,epoch, is_best):
    # save state of the network
    check_point_params = {}
    model = list2sequential(model)
    if isinstance(model, nn.DataParallel):
        check_point_params["model"] = model.module.state_dict()
    else:
        check_point_params["model"] = model.state_dict()

    aux_fc_state = []
    aux_fc_opt_state = []
    seg_opt_state = []
    if aux:
        for i in range(len(model.seg_optimizer)):
            if isinstance(aux[i], nn.DataParallel):
                temp_state = aux[i].module.state_dict()
            else:
                temp_state = aux[i].state_dict()
            aux_fc_state.append(temp_state)
            if aux_optimizer:
                aux_fc_opt_state.append(aux_optimizer[i].state_dict())
            if seg_optimizer:
                seg_opt_state.append(seg_optimizer[i].state_dict())
    check_point_params["aux"] = aux_fc_state
    check_point_params["aux_optimizer"] = aux_fc_opt_state
    check_point_params["seg_optimizer"] = seg_opt_state
    check_point_params["epoch"] = epoch

    filename = os.path.join(args.save_dir, 'checkpoint.{}.{}.pth.tar'.format(args.arch, args.prefix))
    bestname = os.path.join(args.save_dir, 'best.{}.{}.pth.tar'.format(args.arch, args.prefix))
    # checkpoint_save_name = 'checkpoint.{}.{}.pth.tar'.format(args.arch, args.prefix)

    torch.save(check_point_params, filename)
    # torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, bestname)

def print_log(print_string, log):
    print("{}".format(print_string))
    log.write('{}\n'.format(print_string))
    log.flush()
  
  
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(model, epoch):
    """Sets the learning rate"""
    lr = args.lr * (0.1 ** (epoch // 40))
    for i in range(len(model.seg_optimizer)):
        for param_group in model.seg_optimizer[i].param_groups:
            param_group['lr'] = lr

        for param_group in model.aux_optimizer[i].param_groups:
            param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
