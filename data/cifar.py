import sys
sys.path.append('..')
import math
import torch
import torchvision
import numpy as np
from data import common
from utils import path_utils

cifar_mean_std = [[0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]]

class CIFAR10:
    def __init__(self, cfg):
        img_size = 32
        trn_transform = common.get_aug(cfg,img_size,train=True,mean_std=cifar_mean_std)
        db_path = path_utils.get_datasets_dir(cfg.set)
        cfg.logger.info('{} {} {}'.format('ImageNet',cfg.gpu,db_path))
        trn_dataset = torchvision.datasets.CIFAR10(db_path, train=True, transform=trn_transform, download=False)
        self.sampler = torch.utils.data.distributed.DistributedSampler(
            trn_dataset, rank=cfg.gpu, num_replicas=cfg.world_size, shuffle=True
        )

        self.trn_loader = torch.utils.data.DataLoader(
            dataset=trn_dataset,
            batch_size=cfg.batch_size // cfg.world_size,
            num_workers=max(cfg.num_threads // cfg.world_size,1) ,
            # num_workers=0 ,
            shuffle=False,
            pin_memory=True,
            drop_last=True,
            sampler=self.sampler,
        )
        # self.trn_loader.num_batches = math.floor(len(trn_dataset) / (cfg.batch_size * cfg.world_size))
        self.trn_loader.num_batches = math.floor(len(trn_dataset) / (cfg.batch_size))
        self.trn_loader.num_files = len(trn_dataset)
        # self.trn_loader.batch_size = cfg.batch_size

        tst_transform = common.get_aug(cfg, img_size, train=False,mean_std=cifar_mean_std)
        tst_dataset = torchvision.datasets.CIFAR10(db_path, train=False, transform=tst_transform, download=False)
        self.tst_loader = torch.utils.data.DataLoader(
            dataset=tst_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=max(cfg.num_threads//4,1),
            # num_workers=0,
            pin_memory=True,
            drop_last=True
        )
        self.tst_loader.num_batches = math.floor(len(tst_dataset) / cfg.batch_size)
        self.tst_loader.num_files = len(tst_dataset)

        self.val_loader = self.tst_loader
        self.knn_loader = None

class CIFAR100:
    def __init__(self, cfg):
        img_size = 32
        trn_transform = common.get_aug(cfg,img_size,train=True,mean_std=cifar_mean_std)
        db_path = path_utils.get_datasets_dir(cfg.set)
        cfg.logger.info('{} {} {}'.format('ImageNet',cfg.gpu,db_path))
        trn_dataset = torchvision.datasets.CIFAR100(db_path, train=True, transform=trn_transform, download=False)
        self.sampler = torch.utils.data.distributed.DistributedSampler(
            trn_dataset, rank=cfg.gpu-cfg.base_gpu, num_replicas=cfg.world_size, shuffle=True
        )

        self.trn_loader = torch.utils.data.DataLoader(
            dataset=trn_dataset,
            batch_size=cfg.batch_size // cfg.world_size,
            num_workers=max(cfg.num_threads // cfg.world_size,1) ,
            # num_workers=0 ,
            shuffle=False,
            pin_memory=True,
            drop_last=True,
            sampler=self.sampler,
        )
        # self.trn_loader.num_batches = math.floor(len(trn_dataset) / (cfg.batch_size * cfg.world_size))
        self.trn_loader.num_batches = math.floor(len(trn_dataset) / (cfg.batch_size))
        self.trn_loader.num_files = len(trn_dataset)
        # self.trn_loader.batch_size = cfg.batch_size

        tst_transform = common.get_aug(cfg, img_size, train=False,mean_std=cifar_mean_std)
        tst_dataset = torchvision.datasets.CIFAR100(db_path, train=False, transform=tst_transform, download=False)
        self.tst_loader = torch.utils.data.DataLoader(
            dataset=tst_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=max(cfg.num_threads//4,1),
            # num_workers=0,
            pin_memory=True,
            drop_last=True
        )
        self.tst_loader.num_batches = math.floor(len(tst_dataset) / cfg.batch_size)
        self.tst_loader.num_files = len(tst_dataset)

        self.val_loader = self.tst_loader

if __name__ == '__main__':
    class Config:
        def __init__(self):
            pass
    cfg = Config()
    cfg.set = 'CIFAR10'
    cfg.batch_size = 32
    cfg.trn_phase = 'pretrain'
    cfg.arch = 'SimSiam'
    cfg.gpu = 0
    cfg.world_size = 1
    cfg.num_threads = 16
    dataset = CIFAR10(cfg)
    for batch in dataset.val_loader:
        print(batch[0][0,:].mean(axis=[1,2]),batch[0][0,:].std(axis=[1,2]))
        print(batch[0][1,:].mean(axis=[1,2]),batch[0][1,:].std(axis=[1,2]))
        print(batch[0][2,:].mean(axis=[1,2]),batch[0][2,:].std(axis=[1,2]))
        print(batch[0][3,:].mean(axis=[1,2]),batch[0][3,:].std(axis=[1,2]))

        break