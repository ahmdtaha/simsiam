import time
import torch
import numpy as np
import torch.nn as nn
from utils import net_utils
from utils import csv_utils
import torch.nn.functional as F
from sklearn.cluster import KMeans
from utils.logging import AverageMeter, ProgressMeter
from sklearn.metrics import normalized_mutual_info_score


__all__ = ["train", "validate"]




def train(train_loader, model,criterion, optimizer, epoch, cfg, writer):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.3f")
    progress = ProgressMeter(
        train_loader.num_batches,
        [batch_time, data_time, losses],cfg,
        prefix=f"Epoch: [{epoch}]",
    )

    # switch to train mode
    model.train()

    # batch_size = train_loader.batch_size
    num_batches = train_loader.num_batches
    end = time.time()
    batch_size = train_loader.batch_size
    for i , data in enumerate(train_loader):
        imgs1,imgs2, target = data[0][0].cuda(non_blocking=True),data[0][1].cuda(non_blocking=True), data[1].long().squeeze().cuda(non_blocking=True)
        # measure data loading time
        data_time.update(time.time() - end)


        #compute output
        emb1,emb2 = model(imgs1,imgs2)
        loss = criterion(emb1,emb2)

        losses.update(loss.item(), batch_size)
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()


        if i % cfg.print_freq == 0 or i == num_batches-1:
            t = (num_batches * epoch + i) * batch_size
            progress.display(i)
            progress.write_to_tensorboard(writer, prefix="train", global_step=t)




def validate(train_loader,val_loader, model, cfg, writer, epoch,k=200, t=0.1):
    model.eval()
    classes = cfg.num_cls
    total_top1, total_top5, total_num, feature_bank,feature_labels  = 0.0, 0.0, 0, [],[]

    trn_batch_size = train_loader.batch_size
    with torch.no_grad():
        # generate feature bank
        for i, data in enumerate(train_loader):
            imgs1, imgs2, target = data[0][0].cuda(cfg.gpu), data[0][1].cuda(), data[1].long().squeeze()

            feature = model.backbone(imgs1)
            feature = F.normalize(feature, dim=1)
            feature_bank.append(feature.cpu())
            feature_labels.append(target)

        cfg.logger.info('Finish the Trn Features -> Tst Features')
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        # feature_labels = torch.tensor(train_loader.dataset.targets, device=feature_bank.device)
        feature_labels = torch.cat(feature_labels, dim=0)
        # loop test data to predict the label by weighted knn search
        for batch_idx, data in enumerate(val_loader):
            images, target = data[0].cuda(cfg.gpu), data[1].long().squeeze()
            # images, target = data[0]['data'], data[0]['data_aug'], data[0]['label'].long().squeeze()

            feature = model.backbone(images)
            feature = F.normalize(feature, dim=1).cpu()

            pred_labels = knn_predict(feature, feature_bank, feature_labels, classes, k, t)

            total_num += images.size(0)
            total_top1 += (pred_labels[:, 0] == target).float().sum().item()

    cfg.logger.info("Acc {}".format(total_top1 / total_num * 100))
    return total_top1 / total_num * 100

# knn monitor as in InstDisc https://arxiv.org/abs/1805.01978
# implementation follows http://github.com/zhirongw/lemniscate.pytorch and https://github.com/leftthomas/SimCLR
def knn_predict(feature, feature_bank, feature_labels, classes, knn_k, knn_t):
    # compute cos similarity between each feature vector and feature bank ---> [B, N]
    sim_matrix = torch.mm(feature, feature_bank)
    # [B, K]
    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
    # [B, K]
    sim_labels = torch.gather(feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices)
    sim_weight = (sim_weight / knn_t).exp()

    # counts for each class
    one_hot_label = torch.zeros(feature.size(0) * knn_k, classes, device=sim_labels.device)
    # [B*K, C]
    one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
    # weighted score ---> [B, C]
    pred_scores = torch.sum(one_hot_label.view(feature.size(0), -1, classes) * sim_weight.unsqueeze(dim=-1), dim=1)

    pred_labels = pred_scores.argsort(dim=-1, descending=True)
    return pred_labels


