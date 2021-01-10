import os
import sys
import math
import torch
import random
import constants
import trn_pretrain
import logging.config
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
        dist.init_process_group("nccl", rank=rank, world_size=world_size) ## nccl or gloo

def cleanup():
    dist.destroy_process_group()

def spawn_train(cfg):
    # print(torch.cuda.nccl.version())
    # mp.set_start_method("spawn")
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

    model = net_utils.get_model(cfg)
    cfg.logger.info('Moving the model to GPU {}'.format(cfg.gpu))
    model = net_utils.move_model_to_gpu(cfg, model)
    cfg.logger.info('Model conv 1 initialization {}'.format(torch.sum(model.backbone.conv1.weight)))
    if cfg.world_size > 1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(model, device_ids=[cfg.gpu],output_device=cfg.gpu)

    trn_pretrain.trn(cfg, model)

    if cfg.gpu == cfg.base_gpu:
        return_dict['ckpt_path']= None

    cleanup()


def main(arg_num_threads=16,num_gpus=1):
    arg_dataset = 'CIFAR10' # CIFAR10,CIFAR100

    if arg_dataset in ['CIFAR10','CIFAR100']:
        arg_trainer = 'pretrain'
        arg_epochs = str(800)
        arg_test_interval = '1'
        arg_bz = '512'
        arg_backbone = 'resnet18'
        arg_weight_decay = '5e-4'
        arg_lr = str(0.06)
        arg_base_gpu = '0'
    else:
        raise NotImplementedError('Invalid dataset {}'.format(arg_dataset))

    arg_arch = 'SimSiam' # SimCLR, SimSiam


    exp_name_suffix = 'default'
    # exp_name_suffix = 'redo_debug'
    arg_exp_name = 'PRE_{}_{}_{}_lr{}_e{}_bz{}_NG{}_{}/'.format(arg_dataset,
                                               arg_arch,
                                               arg_backbone,
                                               arg_lr,
                                               arg_epochs,
                                               arg_bz,
                                               num_gpus,
                                               exp_name_suffix).replace('.','_')



    if 'debug' in exp_name_suffix:
        arg_num_threads = 0

    arg_dim = '2048' if arg_arch == 'SimSiam' else '512'

    argv = [
        '--name', arg_exp_name,
        '--trn_phase','pretrain',
        '--num_threads', str(arg_num_threads),
        '--base_gpu', arg_base_gpu,

        '--epochs', arg_epochs,
        '--arch', arg_arch,
        '--emb_dim',arg_dim,
        '--backbone',arg_backbone,
        '--world_size', str(num_gpus),
        "--test_interval",arg_test_interval,
        '--save_every',arg_test_interval,
        '--trainer', arg_trainer,

        '--data', constants.dataset_dir ,
        '--set', arg_dataset,

        '--optimizer', 'sgd',
        '--lr', arg_lr,

        '--lr_policy', 'cosine_lr',
        '--warmup_length', '0',

        '--weight_decay', arg_weight_decay,
        '--momentum', '0.9',
        '--batch_size', arg_bz,

    ]

    cfg = Config().parse(argv)

    assert cfg.epochs % 10 == 0 or 'debug' in cfg.name, 'Epoch should be divisible by 10'

    cfg.num_threads = 16

    spawn_train(cfg)


if __name__ == '__main__':
    if len(sys.argv) == 1:
        main()
    else:
        cfg = Config().parse()
        spawn_train(cfg)