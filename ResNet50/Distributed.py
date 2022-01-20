'''Distributed train CIFAR10 with PyTorch.'''
from time import sleep
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import progress_bar

import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

import wandb


# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)

# group settings
world_size = 1                #
DIST_DEFAULT_ADDR = 'localhost'
DIST_DEFAULT_PORT = '12345'
method = f'tcp://{DIST_DEFAULT_ADDR}:{DIST_DEFAULT_PORT}'
backend = 'gloo'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=2, type=int,
                        metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--world_size', default=1, type=int, help='world size')
    args = parser.parse_args()

    #########################################################
    run = wandb.init(
        entity="fahao",
        project="Trace",
        name="Distributed-ResNet50",
        group="DDP",  # all runs for the experiment in one group
    )
    mp.spawn(train, nprocs=args.world_size, args=(args,run,))
    # mp.spawn(train, args=(args, ), nprocs=args.world_size, join=True)
    #########################################################

def train(rank, args, run):
    # CPU or GPU?
    device='cpu'
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        device='cuda'
        print ("use CUDA:", use_cuda, "- device:", device)

    # if rank == 0:
    #     wandb.init(project="Trace", entity="fahao", name="Distributed-ResNet50")

    # init group
    dist.init_process_group(backend=backend,
                                init_method=method,
                                world_size=args.world_size,
                                rank=rank,
                                group_name="DDP")
    if device == 'cuda':
        torch.cuda.set_device(rank)

    # training and test sets
    if rank == 0:
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=128, shuffle=True, num_workers=2)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=100, shuffle=False, num_workers=2)
    else:
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=256, shuffle=True, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck')

    # load model to device
    net = ResNet152()
    net = net.cuda()
    device = torch.device("cuda", rank)
    nn.parallel.DistributedDataParallel(net, device_ids=[rank])
    # if device == 'cuda':
    #     net = torch.nn.DataParallel(net)
    #     cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                        momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    # Train
    for epoch in range (args.epochs):
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            # optimizer.zero_grad()
            # outputs = net(inputs)
            # loss = criterion(outputs, targets)

            # loss.backward()
            # optimizer.step()

            if rank == 1:
                for i in range(10):
                    optimizer.zero_grad()
                    outputs = net(inputs)
                    loss = criterion(outputs, targets)
                    # loss.backward()
                    # optimizer.step()
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
            elif rank == 0:
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            if rank == 0:
                progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
                # sleep(1.5)
        # if rank == 0:
        #     global best_acc
        #     net.eval()
        #     test_loss = 0
        #     correct = 0
        #     total = 0
        #     with torch.no_grad():
        #         for batch_idx, (inputs, targets) in enumerate(testloader):
        #             inputs, targets = inputs.to(device), targets.to(device)
        #             outputs = net(inputs)
        #             loss = criterion(outputs, targets)

        #             test_loss += loss.item()
        #             _, predicted = outputs.max(1)
        #             total += targets.size(0)
        #             correct += predicted.eq(targets).sum().item()

        #             progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))


if __name__ == "__main__":
    main()
