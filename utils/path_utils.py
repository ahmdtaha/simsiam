import os
import getpass
import pathlib
import constants
import os.path as osp

username = getpass.getuser()

def get_checkpoint_dir():

    project_name = osp.basename(osp.abspath('./'))
    ckpt_dir = constants.checkpoint_dir

    assert osp.exists(ckpt_dir),('{} does not exists'.format(ckpt_dir))

    ckpt_dir = f'{ckpt_dir}/{project_name}'
    return ckpt_dir


def get_datasets_dir(dataset_name):


    datasets_dir = constants.dataset_dir

    assert osp.exists(datasets_dir),('{} does not exists'.format(datasets_dir))
    # print(dataset_name)
    if dataset_name == 'CIFAR10':
        dataset_dir = 'cifar10'
    elif dataset_name == 'CIFAR100':
        dataset_dir = 'cifar100'
    else:
        raise NotImplementedError('Invalid dataset name {}'.format(dataset_name))

    datasets_dir = '{}/{}'.format(datasets_dir, dataset_dir)

    return datasets_dir



def get_directories(args,gpu):
    # if args.config_file is None or args.name is None:
    if args.config_file is None and args.name is None:
        raise ValueError("Must have name and config")

    # config = pathlib.Path(args.config_file).stem
    config = args.name
    if args.log_dir is None:
        run_base_dir = pathlib.Path(
            f"{get_checkpoint_dir()}/{args.name}/gpu_{gpu}"
        )
    else:
        run_base_dir = pathlib.Path(
            f"{args.log_dir}/{args.name}"
        )
    
    def _run_dir_exists(run_base_dir):
        log_base_dir = run_base_dir / "logs"
        ckpt_base_dir = run_base_dir / "checkpoints"

        return log_base_dir.exists() or ckpt_base_dir.exists()

   # if _run_dir_exists(run_base_dir):
    rep_count = 0

    # while _run_dir_exists(run_base_dir / '{:04d}'.format(rep_count,args.gpu)):
    #     rep_count += 1

    # date_time_int = int(datetime.now().strftime('%Y%m%d%H%M'))
    run_base_dir = run_base_dir / '{:04d}'.format(rep_count)

    log_base_dir = run_base_dir / "logs"
    ckpt_base_dir = run_base_dir / "checkpoints"

    if not run_base_dir.exists():
        os.makedirs(run_base_dir,exist_ok=True)

    if not run_base_dir.exists():
        os.makedirs(log_base_dir,exist_ok=True)

    (run_base_dir / "settings.txt").write_text(str(args))

    return run_base_dir, ckpt_base_dir, log_base_dir


if __name__ == '__main__':
    print(get_checkpoint_dir('test_exp'))
    # print(get_pretrained_ckpt('vgg_tensorpack'))
    print(get_datasets_dir('cub'))