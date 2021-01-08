import os
import math
import glob
import torch
import shutil
import models
import pathlib
import torch.nn as nn
import torch.backends.cudnn as cudnn




def get_model(cfg):

    cfg.logger.info("=> Creating model '{}'".format(cfg.arch))
    model = models.__dict__[cfg.arch](cfg)

    return model



def move_model_to_gpu(args, model):
    assert torch.cuda.is_available(), "CPU-only experiments currently unsupported"
    # print('{}'.format(args.gpu))
    if args.gpu is not None:
        print('Moving Model to GPU {}'.format(args.gpu))
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    elif args.multigpu is None:
        device = torch.device("cpu")
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        args.logger.info(f"=> Parallelizing on {args.multigpu} gpus")
        torch.cuda.set_device(args.multigpu[0])
        args.gpu = args.multigpu[0]
        model = torch.nn.DataParallel(model, device_ids=args.multigpu).cuda(
            args.multigpu[0]
        )

    cudnn.benchmark = True

    return model


def load_pretrained(pretrained_path,gpus, model,cfg):
    if os.path.isfile(pretrained_path):
        cfg.logger.info("=> loading pretrained weights from '{}'".format(pretrained_path))
        pretrained = torch.load(
            pretrained_path,
            map_location=torch.device("cuda:{}".format(gpus)),
        )["state_dict"]
        skip = ' '
        # skip = 'scores'
        model_state_dict = model.state_dict()
        for k, v in pretrained.items():
            # if k not in model_state_dict or v.size() != model_state_dict[k].size():
            if k not in model_state_dict or v.size() != model_state_dict[k].size() or skip in k:
                cfg.logger.info("IGNORE: {}".format(k))
        pretrained = {
            k: v
            for k, v in pretrained.items()
            if (k in model_state_dict and v.size() == model_state_dict[k].size() and skip not in k)
        }
        model_state_dict.update(pretrained)
        model.load_state_dict(model_state_dict)

    else:
        cfg.logger.info("=> no pretrained weights found at '{}'".format(pretrained_path))


def save_checkpoint(state, is_best, filename="checkpoint.pth", save=False,max_save=3):
    filename = pathlib.Path(filename)

    if not filename.parent.exists():
        os.makedirs(filename.parent,exist_ok=True)

    torch.save(state, filename)

    if is_best:
        shutil.copyfile(filename, str(filename.parent / "model_best.pth"))

        if not save:
            os.remove(filename)
    if max_save > 0:
        all_saved_states = sorted(glob.glob(str(filename.parent)+'/epoch_*.state'),key=os.path.getmtime)
        # if len(all_saved_states) > 1:
        for i in range(len(all_saved_states)-max_save): ## Keeping one state only to avoid vulcan memory issues
            os.remove(all_saved_states[i])

# def save_checkpoint(state, is_best, filename="checkpoint.pth", save=False):
#     filename = pathlib.Path(filename)
#
#     if not filename.parent.exists():
#         os.makedirs(filename.parent)
#
#     torch.save(state, filename)
#
#     if is_best:
#         shutil.copyfile(filename, str(filename.parent / "model_best.pth"))
#
#         if not save:
#             os.remove(filename)


def get_lr(optimizer):
    return optimizer.param_groups[0]["lr"]




def accumulate(model, f):
    acc = 0.0

    for child in model.children():
        acc += accumulate(child, f)

    acc += f(model)

    return acc




class LabelSmoothing(nn.Module):
    """
    NLL loss with label smoothing.
    """

    def __init__(self, smoothing=0.0):
        """
        Constructor for the LabelSmoothing module.

        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class SubnetL1RegLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, model, temperature=1.0):
        l1_accum = 0.0
        for n, p in model.named_parameters():
            if n.endswith("mask"):
                l1_accum += (p*temperature).sigmoid().sum()

        return l1_accum


