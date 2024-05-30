import os
import sys
import argparse
import torch
import torch.backends.cudnn as cudnn
from torchvision import models

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.model.network import sNet, SimCLR, ParamRegression
from src.model.hair_dataset import HairDataset, HairImageDataset

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch SimCLR')
parser.add_argument('--data', metavar='DIR',  required=True, help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 12)')
parser.add_argument('--epochs', default=5000, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=32, type=int, metavar='N',
                    help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.0003, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--fp16-precision', default=True, type=bool,
                    help='Whether or not to use 16-bit precision GPU training.')

parser.add_argument('--out_dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--crop_size', default=0, type=int,
                    help='crop size of training images (default: 0)')
parser.add_argument('--same_crop', default=True, type=bool,
                    help='Whether or not to use the same crop for both images in a pair.')
parser.add_argument('--use_depth', default=True, type=bool,
                    help='Whether or not to use depth image.')
parser.add_argument('--use_orien', default=True, type=bool,
                    help='Whether or not to use orientation image.')
parser.add_argument('--use_param', default=True, type=bool,
                    help='Whether or not to use mask image.')

parser.add_argument('--log-every-n-epoch', default=10, type=int,
                    help='Log every n epoch')
parser.add_argument('--temperature', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')
parser.add_argument('--gpu-index', default=0, type=int, help='Gpu index.')
parser.add_argument('--comment', default='', type=str, help='Comment for tensorboard.')


def main():
    args = parser.parse_args()
    args.device = torch.device('cuda')
    print("==========> args <==========")
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    print("============================\n")
    cudnn.deterministic = True
    cudnn.benchmark = True

    train_dataset = HairDataset(
        os.path.join(args.data, "train"),
        use_orien=args.use_orien,
        use_depth=args.use_depth,
        use_param=args.use_param,
        augment=True,
        crop_size=args.crop_size,
        same_crop=args.same_crop,
    )
    test_dataset = HairDataset(
        os.path.join(args.data, "test"),
        use_orien=args.use_orien,
        use_depth=args.use_depth,
        use_param=args.use_param,
        augment=False,
        crop_size=args.crop_size,
        same_crop=args.same_crop,
    )

    print(f"Dataset size | train:{len(train_dataset)}, test:{len(test_dataset)}")

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, persistent_workers=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=True, persistent_workers=True,
        num_workers=1, pin_memory=True, drop_last=False)

    in_channels = [1, args.use_orien*2 + args.use_depth + args.use_param]
    model = sNet(base_model=args.arch, out_dim=args.out_dim, in_channels=in_channels)

    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0,
                                                           last_epoch=-1)

    #  It’s a no-op if the 'gpu_index' argument is a negative integer or None.
    with torch.cuda.device(args.gpu_index):
        simclr = SimCLR(model=model, optimizer=optimizer, scheduler=scheduler, args=args)
        simclr.train(train_loader, test_loader)


def regression():
    args = parser.parse_args()
    args.device = torch.device('cuda')
    print("==========> args <==========")
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    print("============================\n")
    cudnn.deterministic = True
    cudnn.benchmark = True

    train_dataset = HairImageDataset(
        os.path.join(args.data, "train"),
        augment=True,
    )
    test_dataset = HairImageDataset(
        os.path.join(args.data, "test"),
        augment=False,
    )

    print(f"Dataset size | train:{len(train_dataset)}, test:{len(test_dataset)}")

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, persistent_workers=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=True, persistent_workers=True,
        num_workers=1, pin_memory=True, drop_last=False)

    model = sNet(base_model=args.arch, out_dim=1, in_channels=[1, 1])

    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0,
                                                           last_epoch=-1)

    #  It’s a no-op if the 'gpu_index' argument is a negative integer or None.
    with torch.cuda.device(args.gpu_index):
        regression = ParamRegression(model=model, optimizer=optimizer, scheduler=scheduler, args=args)
        regression.train(train_loader, test_loader)


if __name__ == "__main__":
    main()
    # regression()
