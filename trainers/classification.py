import time
import torch
from utils.eval_utils import accuracy
from utils.logging import AverageMeter, ProgressMeter



__all__ = ["train", "validate"]




def train(train_loader, model, criterion, optimizer, epoch, cfg, writer):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.3f")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    progress = ProgressMeter(
        train_loader.num_batches,
        [batch_time, data_time, losses, top1, top5],cfg,
        prefix=f"Epoch: [{epoch}]",
    )
    # switch to train mode
    model.train()
    batch_size = train_loader.batch_size
    num_batches = train_loader.num_batches
    end = time.time()
    for i , data in enumerate(train_loader):
        images, target = data[0].cuda(),data[1].long().squeeze().cuda()
        # measure data loading time
        data_time.update(time.time() - end)

        #compute output
        output = model(images)
        loss = criterion(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))


        # measure accuracy and record loss
        losses.update(loss.item(), batch_size)
        top1.update(acc1.item(), batch_size)
        top5.update(acc5.item(), batch_size)

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


    return top1.avg, top5.avg


def validate(val_loader, model, criterion, args, writer, epoch):
    batch_time = AverageMeter("Time", ":6.3f", write_val=False)
    losses = AverageMeter("Loss", ":.3f", write_val=False)
    top1 = AverageMeter("Acc@1", ":6.2f", write_val=False)
    top5 = AverageMeter("Acc@5", ":6.2f", write_val=False)
    progress = ProgressMeter(
        val_loader.num_batches, [batch_time, losses, top1, top5],args, prefix="Test: "
    )
    # switch to evaluate mode
    model.eval()
    batch_size = val_loader.batch_size
    with torch.no_grad():
        end = time.time()
        for i, data in enumerate(val_loader):
            images, target = data[0].cuda(), data[1].long().squeeze().cuda()
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            losses.update(loss.item(), batch_size)
            top1.update(acc1.item(), batch_size)
            top5.update(acc5.item(), batch_size)


            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()



            if i % args.print_freq == 0:
                progress.display(i)

        progress.display(val_loader.num_batches)

        if writer is not None:
            progress.write_to_tensorboard(writer, prefix="test", global_step=epoch)

    return top1.avg, top5.avg
