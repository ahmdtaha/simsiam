import os
import sys
import yaml
import argparse
import os.path as osp
import logging.config
from utils import os_utils
from utils import log_utils
from utils import path_utils
# from configs import parser as _parser

args = None

class Config:
    def __init__(self):
        parser = argparse.ArgumentParser(description="Self-supervised baseline")

        # General Config
        parser.add_argument(
            "--data", help="path to dataset base directory", default="/mnt/disk1/datasets"
        )

        parser.add_argument(
            "--trn_phase", help="Which phase is this? pretrain | classification", default="pretrain",
            choices=['pretrain','classification']
        )

        parser.add_argument("--optimizer", help="Which optimizer to use", default="sgd")
        parser.add_argument("--set", help="only CIFAR10 is currently supported", type=str, default="CIFAR10",
                            choices=['CIFAR10','CIFAR100'])

        parser.add_argument(
            "--arch", metavar="ARCH", default="SimSiam", help="model architecture",
            choices=['SimSiam','SimCLR']
        )

        parser.add_argument(
            "--backbone", default="resnet18", help="model architecture",
            choices=['resnet18','resnet50']
        )

        parser.add_argument(
            "--config_file", help="Config file to use (see configs dir)", default=None
        )
        parser.add_argument(
            "--log-dir", help="Where to save the runs. If None use ./runs", default=None
        )



        parser.add_argument(
            "-t",
            "--num_threads",
            default=8,
            type=int,
            metavar="N",
            help="number of data loading workers (default: 20)",
        )
        parser.add_argument(
            "--epochs",
            default=90,
            type=int,
            metavar="N",
            help="number of total epochs to run",
        )

        parser.add_argument(
            "--start-epoch",
            default=None,
            type=int,
            metavar="N",
            help="manual epoch number (useful on restarts)",
        )
        parser.add_argument(
            "-b",
            "--batch_size",
            default=256,
            type=int,
            metavar="N",
            help="mini-batch size (default: 256), this is the total "
                 "batch size of all GPUs on the current node when "
                 "using Data Parallel or Distributed Data Parallel",
        )
        parser.add_argument(
            "--lr",
            "--learning-rate",
            default=0.1,
            type=float,
            metavar="LR",
            help="initial learning rate",
            dest="lr",
        )
        parser.add_argument(
            "--warmup_length", default=0, type=int, help="Number of warmup iterations"
        )
        parser.add_argument(
            "--momentum", default=0.9, type=float, metavar="M", help="momentum"
        )
        parser.add_argument(
            "--wd",
            "--weight_decay",
            default=1e-4,
            type=float,
            metavar="W",
            help="weight decay (default: 1e-4)",
            dest="weight_decay",
        )
        parser.add_argument(
            "-p",
            "--print_freq",
            default=1000,
            type=int,
            metavar="N",
            help="print frequency (default: 10)",
        )
        parser.add_argument('--emb_dim', default=2048, type=int,
                            help='Size of embedding that is appended to backbone model.'
                            )

        parser.add_argument('--l2-norm', default=1, type=int,
                            help='L2 normlization'
                            )
        parser.add_argument('--warm', default=1, type=int,
                            help='Warmup training epochs'
                            )
        parser.add_argument(
            "--resume",
            default="",
            type=str,
            metavar="PATH",
            help="path to latest checkpoint (default: none)",
        )

        parser.add_argument(
            "--pretrained",
            dest="pretrained",
            default=None,
            type=str,
            help="use pre-trained model",
        )
        parser.add_argument(
            "--seed", default=None, type=int, help="seed for initializing training. "
        )


        parser.add_argument(
            "--world_size",
            default=1,
            type=int,
            help="Pytorch DDP rank",
        )

        parser.add_argument(
            "--gpu",
            default=0,
            type=int,
            help="Which GPUs to use?",
        )
        parser.add_argument(
            "--base_gpu",
            default=0,
            type=int,
            help="Which GPUs to use?",
        )
        parser.add_argument(
            "--test_interval", default=10, type=int, help="Eval on tst/val split every ? epochs"
        )

        # Learning Rate Policy Specific
        parser.add_argument(
            "--lr_policy", default="constant_lr", help="Policy for the learning rate."
        )
        parser.add_argument(
            "--multistep-lr-adjust", default=30, type=int, help="Interval to drop lr"
        )
        parser.add_argument("--multistep-lr-gamma", default=0.1, type=int, help="Multistep multiplier")
        parser.add_argument(
            "--name", default=None, type=str, help="Experiment name to append to filepath"
        )
        parser.add_argument(
            "--log_file", default='train_log.txt', type=str, help="Experiment name to append to filepath"
        )
        parser.add_argument(
            "--save_every", default=-1, type=int, help="Save every ___ epochs"
        )

        parser.add_argument('--lr-decay-step', default=10, type=int,help='Learning decay step setting')
        parser.add_argument('--lr-decay-gamma', default=0.5, type=float,help='Learning decay gamma setting')

        parser.add_argument(
            "--trainer", type=str, default="pretrain", help="How to train a model"
        )

        self.parser = parser

    def parse(self,args):
        self.cfg = self.parser.parse_args(args)

        if self.cfg.set == 'CIFAR10':
            self.cfg.num_cls = 10
            self.cfg.eval_tst = True
        elif self.cfg.set == 'CIFAR100':
            self.cfg.num_cls = 100
            self.cfg.eval_tst = True
        else:
            raise NotImplementedError('Invalid dataset {}'.format(self.cfg.set))

        self.cfg.exp_dir = osp.join(path_utils.get_checkpoint_dir() , self.cfg.name)

        os_utils.touch_dir(self.cfg.exp_dir)
        log_file = os.path.join(self.cfg.exp_dir, self.cfg.log_file)
        logging.config.dictConfig(log_utils.get_logging_dict(log_file))
        self.cfg.logger = logging.getLogger('train')

        return self.cfg

