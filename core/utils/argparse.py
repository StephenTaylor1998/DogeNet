import argparse
# import torchvision.models as models
from core import models


def arg_parse():
    model_names = sorted(name for name in models.__dict__
                         if name.islower() and not name.startswith("__")
                         and callable(models.__dict__[name]))

    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--data-path', metavar='DIR', help='path to dataset', default='./data/dataset')
    parser.add_argument('-d', '--data_format', metavar='DATA_FORMAT', default='imagefolder',
                        choices=['image_folder', 'imagenet', 'cifar10', 'cifar100', 'mnist',
                                 'fashion_mnist'],
                        help='data format: (default: imagefolder)')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                        choices=model_names,
                        help='model architecture: ' +
                             ' | '.join(model_names) +
                             ' (default: resnet18)')
    # parser.add_argument('--in-channels', default=3, type=int, metavar='N',
    #                     help='number of classes (default: 3)')
    parser.add_argument('-c', '--classes', default=1001, type=int, metavar='N',
                        help='number of classes (default: 1001)')
    parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')
    parser.add_argument('--epochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=1024, type=int,
                        metavar='N',
                        help='mini-batch size (default: 2048), this is the total '
                             'batch size of all GPUs on the current node when '
                             'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('-t', '--test', dest='test', action='store_true',
                        help='evaluate model on test set')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')
    parser.add_argument('--world-size', default=1, type=int,  # -1
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=0, type=int,  # -1
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:8889', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('--multiprocessing-distributed', action='store_true', default='True',
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')

    # add on args
    parser.add_argument('--in-shape', default=(3, 32, 32), nargs='+', type=int,
                        help='Input image.')
    return parser
