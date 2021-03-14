import argparse
import os
import logging
import pickle
import warnings

warnings.filterwarnings('ignore')

import yaml
from easydict import EasyDict
from torch import nn
from torch.nn import init

log = logging.getLogger(__name__)
from torch.utils.tensorboard import SummaryWriter
from torch.optim import*
#from src.models import *
from ssd import*
from src.fed_zoo import*
#from src.utils import *
from src.utils import*
from data import *


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def parse_args():
    parser = argparse.ArgumentParser(
        description='Single Shot MultiBox Detector Training With Pytorch')
    train_set = parser.add_mutually_exclusive_group()
    parser.add_argument('--dataset', default='VOC', choices=['VOC', 'COCO'],
                        type=str, help='VOC or COCO')
    parser.add_argument('--dataset_root', default=VOC_ROOT,
                        help='Dataset root directory path')
    parser.add_argument('--basenet', default='vgg16_reducedfc.pth',
                        help='Pretrained base model')
    parser.add_argument('--batch_size', default=32, type=int,
                        help='Batch size for training')
    parser.add_argument('--resume', default=None, type=str,
                        help='Checkpoint state_dict file to resume training from')
    parser.add_argument('--start_iter', default=0, type=int,
                        help='Resume training at this iter')
    parser.add_argument('--num_workers', default=4, type=int,
                        help='Number of workers used in dataloading')
    parser.add_argument('--cuda', default=True, type=str2bool,
                        help='Use CUDA to train model')
    parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                        help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='Momentum value for optim')
    parser.add_argument('--weight_decay', default=5e-4, type=float,
                        help='Weight decay for SGD')
    parser.add_argument('--gamma', default=0.1, type=float,
                        help='Gamma update for SGD')
    parser.add_argument('--visdom', default=False, type=str2bool,
                        help='Use visdom for loss visualization')
    parser.add_argument('--save_folder', default='weights/',
                        help='Directory for saving checkpoint models')
    parser.add_argument('-c', '--config', default='./config/configs.yaml', type=str, metavar='Path',
                        help='path to the config file (default: configs.yaml)')

    args = parser.parse_args()

    # with open(args.config) as f:
    with open(os.path.join(os.path.abspath('.'), 'config', 'config.yaml')) as f:
        config = EasyDict(yaml.load(f))

    # Add args parameters to the dict
    for k, v in vars(args).items():
        config[k] = v

    config['save_folder'] = os.path.join(os.path.abspath('.'), config['save_folder'])

    return config


def main():
    os.chdir(os.path.abspath('.'))
    args = parse_args()
    seed_everything(args.seed)
    log.info(args)

    if args.cuda:
        if torch.cuda.is_available():
            if args.cuda:
                torch.set_default_tensor_type('torch.cuda.FloatTensor')
            if not args.cuda:
                print("WARNING: It looks like you have a CUDA device, but aren't " +
                      "using CUDA.\nRun with --cuda for optimal training speed.")
                torch.set_default_tensor_type('torch.FloatTensor')
        else:
            torch.set_default_tensor_type('torch.FloatTensor')

    # 创建 ssd 模型
    if args.dataset == "VOC":
        ssd_cfg = voc
    else:
        ssd_cfg = coco

    model = build_ssd('train', ssd_cfg['min_dim'], ssd_cfg['num_classes'])
    model.extras.apply(weights_init)
    model.loc.apply(weights_init)
    model.conf.apply(weights_init)

    writer = SummaryWriter(log_dir=os.path.join(args.savedir, "tf"))

    # 创建fed
    federater = FedAvg(model=model,
                       optimizer=SGD,
                       #optimizer_args=args.optimizer_args,
                       optimizer_args={
                           'lr':1e-3,
                           'momentum': 0.9,
                           'weight_decay': 5e-4,
                       },
                       num_clients=args.n_client,
                       local_epoch=args.local_epoch,
                       iid=args.iid,
                       device=args.device,
                       writer=writer, args=args)

    federater.fit(args.n_round)

    with open(os.path.join(args.savedir, "result.pkl"), "wb") as f:
        pickle.dump(federater.result, f)


# model init to do

def xavier(param):
    init.xavier_uniform(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()


if __name__ == "__main__":
    main()
