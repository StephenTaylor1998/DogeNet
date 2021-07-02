import os
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from core import models
from core.datasets.data.image_folder import classify_train_dataset
from core.datasets.data.image_folder import classify_val_dataset
from core.datasets.data.image_folder import classify_test_dataset
from core.engine.base import validate, adjust_learning_rate, train, save_checkpoint
from core.utils.copy_weights import copy_weights
from core.utils.resume import resume_from_checkpoint

best_acc1 = 0


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True, num_classes=args.classes)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch](num_classes=args.classes)

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    # optimizer = torch.optim.Adam(model.parameters(), args.lr,
    #                             weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        args, model, optimizer, best_acc1 = resume_from_checkpoint(args, model, optimizer, best_acc1)

    cudnn.benchmark = True

    # Data loading code
    # traindir = os.path.join(args.data, 'train')
    # valdir = os.path.join(args.data, 'val')
    # testdir = os.path.join(args.data, 'test')

    # train data loader is here, distribute is support #
    train_dataset = classify_train_dataset(args.data)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ #

    # val data loader is here #
    val_loader = torch.utils.data.DataLoader(
        classify_val_dataset(args.data),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    # ^^^^^^^^^^^^^^^^^^^^^^^ #
    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    if args.test:
        test_loader = torch.utils.data.DataLoader(
            classify_test_dataset(args.data),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)

        validate(train_loader, model, criterion, args)
        print('TEST IN TRAIN SET')
        validate(val_loader, model, criterion, args)
        print('TEST IN VAL SET')
        validate(test_loader, model, criterion, args)
        print('TEST IN TEST SET')
        return

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, args)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                    and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
            }, is_best, args)
            copy_weights(args, epoch)
