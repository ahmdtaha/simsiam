import os
import sys
import torch
import random
import getpass
import trn_classifier
import logging.config
import torch.nn as nn
from utils import net_utils
from utils import log_utils
import torch.distributed as dist
import torch.multiprocessing as mp
from config.base_config import Config
from torch.nn.parallel import DistributedDataParallel as DDP

start_port = random.choice(range(12355,12375)) #12358 # ResNet 18

def setup(rank, world_size,port):
    if sys.platform == 'win32':
        # Distributed package only covers collective communications with Gloo
        # backend and FileStore on Windows platform. Set init_method parameter
        # in init_process_group to a local file.
        # Example init_method="file:///f:/libtmp/some_file"
        init_method="file:///{your local file path}"

        # initialize the process group
        dist.init_process_group(
            "gloo",
            init_method=init_method,
            rank=rank,
            world_size=world_size
        )
    else:
        print('localhost & port {} & rank {}'.format(port,rank))
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = str(port)

        # initialize the process group
        dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def spawn_train(cfg):
    manager = mp.Manager()
    return_dict = manager.dict()
    jobs = []
    for i in range(cfg.world_size):
        p = mp.Process(target=train_ddp, args=(i,cfg, return_dict))
        jobs.append(p)
        p.start()

    for proc in jobs:
        proc.join()

    return return_dict

def train_ddp(rank,cfg,return_dict):
    cfg.gpu = rank + cfg.base_gpu
    print(f"Train Running basic DDP example on rank {rank}.")
    setup(rank, cfg.world_size, start_port)

    cfg.log_file = 'train_{}.txt'.format(cfg.gpu)
    log_file = os.path.join(cfg.exp_dir, cfg.log_file)
    logging.config.dictConfig(log_utils.get_logging_dict(log_file, mode='a+'))
    cfg.logger = logging.getLogger('train')

    cfg.logger.info('Getting the model')
    pretrain_model = net_utils.get_model(cfg)
    pretrain_model = torch.nn.DataParallel(pretrain_model)
    if cfg.pretrained and cfg.pretrained != 'imagenet':
        net_utils.load_pretrained(cfg.pretrained, cfg.gpu, pretrain_model, cfg)


    classifier_layer = nn.Linear(in_features=pretrain_model.module.backbone.output_dim, out_features=cfg.num_cls,
                                 bias=True).cuda()

    for m in pretrain_model.parameters():
        if hasattr(m, "requires_grad") and m.requires_grad is not None:
            m.requires_grad = False
    cfg.logger.info(
        'Start Training: Model conv 1 initialization {}'.format(torch.sum(pretrain_model.module.backbone.conv1.weight)))
    model = nn.Sequential(
        pretrain_model.module.backbone,
        classifier_layer,
    )

    cfg.logger.info('Moving the model to GPU {}'.format(cfg.gpu))
    model = net_utils.move_model_to_gpu(cfg, model)
    cfg.logger.info('Model conv 1 initialization {}'.format(torch.sum(model[0].conv1.weight)))
    if cfg.world_size > 1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(model, device_ids=[cfg.gpu],output_device=cfg.gpu)

    trn_classifier.trn(cfg,model)
    if cfg.gpu == cfg.base_gpu:
        return_dict['ckpt_path']= None

    cleanup()

def main(arg_num_threads=8,num_gpus=1):
    arg_dataset = 'CIFAR10'
    arg_epochs = str(90)
    arg_arch = 'SimSiam'
    arg_backbone = 'resnet18'

    arg_pretrained_model = '/mnt/data/checkpoints/simsiam/PRE_CIFAR10_SimSiam_resnet18_lr0_06_e800_bz512_NG4_default/gpu_0/0000/checkpoints/epoch_0799.state'
    assert os.path.exists(arg_pretrained_model), 'Please provide a valid pretrained model'
    exp_name_suffix = arg_pretrained_model.split('/')[5]
    arg_exp_name = 'CLS_{}_{}_e{}_{}/'.format(arg_dataset,
                                                               arg_arch,
                                                               arg_epochs,
                                                               exp_name_suffix)

    arg_weight_decay = '0'

    if 'debug' in exp_name_suffix:
        arg_num_threads = 0


    argv = [
        '--name', arg_exp_name,
        '--trn_phase', 'classification',
        '--pretrained',arg_pretrained_model,
        '--num_threads', str(arg_num_threads),
        '--gpu', '0',
        '--epochs', arg_epochs,
        '--arch', arg_arch,
        '--backbone',arg_backbone,
        '--world_size', str(num_gpus),
        "--test_interval",'2',
        '--save_every','2',
        '--trainer', 'classification',

        '--data', '/mnt/data/datasets/',
        '--set', arg_dataset,  # Flower102, CUB200
        '--weight_decay', arg_weight_decay,
        '--momentum', '0.9',

        # '--optimizer', 'lars',
        # '--lr', '0.32',
        # '--batch_size', '4096',

        '--optimizer', 'sgd',
        '--lr', '30',
        '--batch_size', '256',

        # '--lr_policy', 'step_lr',
        # '--warmup_length', '5',

        '--lr_policy', 'cosine_lr',
        # '--warmup_length', '10',

    ]

    cfg = Config().parse(argv)

    assert cfg.epochs % 10 == 0 or 'debug' in cfg.name, 'Epoch should be divisible by 10'

    spawn_train(cfg)


if __name__ == '__main__':
    if len(sys.argv) == 1:
        main()
    else:
        cfg = Config().parse()
        spawn_train(cfg)