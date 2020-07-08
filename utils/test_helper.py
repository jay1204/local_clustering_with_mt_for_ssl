import logging
import torch
from torch.autograd import Variable

from utils.train_helper import *


def mt_test(test_loader, model):
    cross_entropy_loss = torch.nn.CrossEntropyLoss().cuda()

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.eval()
    for i, (x, gt) in enumerate(test_loader):
        batch_size = x.size(0)

        x = Variable(x)
        gt = Variable(gt.type(torch.cuda.LongTensor), requires_grad=False)

        # compute output
        with torch.no_grad():
            _, logits = model(x)
            loss = cross_entropy_loss(logits, gt)

        # accuracy
        prec1, prec5 = accuracy(logits.data, gt.data, topk=(1, 5))

        losses.update(loss.item(), batch_size)
        top1.update(prec1[0], batch_size)
        top5.update(prec5[0], batch_size)

    logging.info(('Testing Results: Prec@1 {top1.avg:.4f} Prec@5 {top5.avg:.4f} Loss {loss.avg:.4f}'
                  .format(top1=top1, top5=top5, loss=losses)))

    return top1.avg

