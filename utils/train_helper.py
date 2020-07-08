import torch
import shutil
import numpy as np
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_
from torch.cuda import FloatTensor
import torch.nn.functional as F
import matplotlib
import pickle


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Record(object):
    """Records model performance"""
    def __init__(self, record_name):
        self.record_name = record_name
        self.accs = []
        self.epochs = []

    def update(self, acc, epoch):
        self.accs.append(acc)
        self.epochs.append(epoch)


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def linear_decay(decay_start, decay_end, epoch, optimizer, init_lr):
    if decay_end <= decay_start:
        return

    if epoch < decay_start:
        return

    lr = init_lr * (decay_end - epoch) / (decay_end - decay_start)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return


def save_checkpoint(model, prefix, filename='checkpoint.pth.tar'):
    filename = '{}'.format(prefix) + filename
    state = {'state_dict': model.state_dict()}
    torch.save(state, filename)


def load_checkpoint(prefix, filename='checkpoint.pth.tar'):
    filename = '{}'.format(prefix) + filename

    checkpoint = torch.load(filename)
    return checkpoint['state_dict']


def init_weights(net):
    """Initialize network weights

    :param net:
    :return:
    """
    def init_func(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.05)
            if m.bias is not None:
                m.bias.data.zero_()
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif classname.find('Linear') != -1:
            if m.bias is not None:
                m.bias.data.zero_()

    net.apply(init_func)


def param_init(model, data_loader):
    def func_gen(flag):
        def func(m):
            if hasattr(m, 'init_mode'):
                setattr(m, 'init_mode', flag)

        return func

    model.apply(func_gen(True))
    data_iter = iter(data_loader)
    (x, _), _ = next(data_iter)
    _ = model(Variable(x).cuda())
    model.apply(func_gen(False))

    return


def update_ema_variables(model, ema_model, ema_decay=0.9999):
    """Update ema_model param as the exponential moving average of model,"""
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(ema_decay).add_(1-ema_decay, param.data)

    return


def sigmoid_rampup(epoch, total_epochs):
    """exponential ramp up the multiplier from 0 to 1 in early training stage"""
    if epoch >= total_epochs:
        return 1.0
    elif epoch <= 0:
        return 0.0
    else:
        phase = 1.0 - float(epoch) / total_epochs
        return float(np.exp(-5.0 * phase * phase))


def linear_rampup(epoch, total_epochs):
    """linear up the multiplier from 0 to 1 in early training stage"""
    if epoch >= total_epochs:
        return 1.0
    else:
        return float(epoch) / total_epochs


def softmax_mse_loss(logits1, logits2):
    """Take softmax on both and return mse loss on top"""
    assert logits1.size() == logits2.size()
    softmax1 = F.softmax(logits1, dim=1)
    softmax2 = F.softmax(logits2, dim=1)
    return F.mse_loss(softmax1, softmax2)


def mse_loss(f1, f2):
    assert f1.size() == f2.size()
    return F.mse_loss(f1, f2)


def softmax_l1_loss(logits1, logits2):
    softmax1 = F.softmax(logits1, dim=1)
    softmax2 = F.softmax(logits2, dim=1)
    return F.l1_loss(softmax1, softmax2)


def pickle_save_record(record, file_name):
    with open(file_name + '.pkl', 'wb') as output:
        pickle.dump(record, output, pickle.HIGHEST_PROTOCOL)


def pickle_read_record(file_name):
    with open(file_name + '.pkl', 'rb') as input:
        return pickle.load(input)