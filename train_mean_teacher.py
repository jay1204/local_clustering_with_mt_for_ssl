import logging
import time
import torch
import torchvision
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torch.cuda import FloatTensor
from torch.cuda import LongTensor

from model.mean_teacher_arch import ShallowNet
from utils.data_helper import *
from utils.data_loader import ImageIter, TransformTwice
from utils.train_helper import *
from utils.test_helper import *
from config import cifar10_params, model_training, svhn_params


def main():
    if model_training.dataset == 'cifar10':
        data_params = cifar10_params
    elif model_training.dataset == 'svhn':
        data_params = svhn_params
    else:
        NotImplementedError('The parameters for this dataset {} not set up yet.'.format(model_training.dataset))

    if not os.path.exists(data_params.save_dir):
        os.makedirs(data_params.save_dir)

    if not os.path.exists(data_params.log_dir):
        os.makedirs(data_params.log_dir)

    logging.basicConfig(filename=data_params.log_dir + 'train-mt-{}.log'.format(model_training.model_id),
                        level=logging.INFO)
    logging.info('Start training model with {} dataset. Time: {}'.
                 format(model_training.dataset, time.asctime(time.localtime(time.time()))))

    logging.info('Model params: {}'.format(model_training))
    logging.info('Training params: {}'.format(data_params))

    def create_model(ema=False):
        model = ShallowNet(num_classes=data_params.num_classes)
        model = torch.nn.DataParallel(model, device_ids=data_params.device_ids).cuda()
        init_weights(model)

        if ema:
            for param in model.parameters():
                param.detach_()

        return model

    model = create_model()
    ema_model = create_model(ema=True)

    cudnn.benchmark = True

    if model_training.dataset == 'cifar10':
        train_label_iter, train_unlabel_iter, test_iter = create_train_test_split(
            train_data_label_percentage=data_params.train_data_label_percentage,
            label_random_seed=data_params.label_random_seed, dataset=model_training.dataset)
    elif model_training.dataset == 'svhn':
        train_label_iter, train_unlabel_iter, test_iter = create_train_test_split_by_quantity(
            train_num_label_examples=data_params.train_num_label_examples,
            label_random_seed=data_params.label_random_seed,
            dataset=model_training.dataset)
    else:
        NotImplementedError('The data splitting func for this dataset {} not set up yet.'.format(model_training.dataset))

    train_label_loader = torch.utils.data.DataLoader(train_label_iter, batch_size=data_params.label_batch_size,
                                                     shuffle=True, num_workers=data_params.num_workers,
                                                     pin_memory=True, drop_last=True)

    train_unlabel_loader = torch.utils.data.DataLoader(train_unlabel_iter, batch_size=data_params.batch_size,
                                                       shuffle=True, num_workers=data_params.num_workers,
                                                       pin_memory=True, drop_last=True)

    test_loader = torch.utils.data.DataLoader(test_iter, batch_size=data_params.batch_size, shuffle=False,
                                              num_workers=data_params.num_workers, pin_memory=True)

    if data_params.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=data_params.learning_rate, betas=(0.5, 0.999))
    elif data_params.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=data_params.learning_rate, momentum=0.9,
                                    weight_decay=0.0001, nesterov=True)
    else:
        raise NotImplementedError('Optimizer {} is not supported yet.'.format(data_params.optimizer))

    # Data-dependent initialization for weight norm layer
    param_init(model, train_label_loader)
    param_init(ema_model, train_label_loader)

    # Training
    best_ema_valid_acc = 0.0
    best_ema_epoch = 1
    train_record = Record('train_accuracy')
    valid_record = Record('valid_accuracy')
    ema_valid_record = Record('ema_valid_accuracy')

    for epoch in range(1, data_params.schedule_epoch[-1] + 1):
        linear_decay(data_params.schedule_epoch[0], data_params.schedule_epoch[-1], epoch, optimizer,
                     init_lr=data_params.learning_rate)
        train_acc = train(train_label_loader, train_unlabel_loader, model, ema_model, optimizer, epoch, data_params)
        train_record.update(train_acc, epoch)

        if epoch > 0 and epoch % data_params.valid_freq == 0:
            logging.info('Evaluation on model:')
            valid_acc = mt_test(test_loader, model)
            valid_record.update(valid_acc, epoch)
            logging.info('Evaluation on EMA model: ')
            ema_valid_acc = mt_test(test_loader, ema_model)
            ema_valid_record.update(ema_valid_acc, epoch)
            if ema_valid_acc > best_ema_valid_acc:
                best_ema_valid_acc = ema_valid_acc
                best_ema_epoch = epoch
                save_checkpoint(model, prefix=data_params.save_dir + 'model_{}'.format(model_training.model_id))
                save_checkpoint(ema_model, prefix=data_params.save_dir + 'model_{}_ema'.format(model_training.model_id))

    logging.info('The best trained is in epoch {} with validation acc {}'.format(best_ema_epoch, best_ema_valid_acc))
    pickle_save_record(train_record,
                       file_name=data_params.save_dir + 'model_{}_train_record'.format(model_training.model_id))
    pickle_save_record(valid_record,
                       file_name=data_params.save_dir + 'model_{}_valid_record'.format(model_training.model_id))
    pickle_save_record(ema_valid_record,
                       file_name=data_params.save_dir + 'model_{}_ema_valid_record'.format(model_training.model_id))

    return


def train(train_label_loader, train_unlabel_loader, model, ema_model, optimizer, epoch, data_params):
    train_label_iter = iter(train_label_loader)

    # Define the loss terms
    cross_entropy_loss = torch.nn.CrossEntropyLoss().cuda()

    # Define record terms for primary model
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # Define record terms for EMA model
    ema_losses = AverageMeter()
    ema_top1 = AverageMeter()
    ema_top5 = AverageMeter()

    # Define consistency loss record term
    consistency_losses = AverageMeter()

    model.train()
    ema_model.train()
    for i, ((unlabel_x, ema_unlabel_x), _) in enumerate(train_unlabel_loader):

        try:
            (label_x, ema_label_x), label_gt = next(train_label_iter)
        except StopIteration:
            train_label_iter = iter(train_label_loader)
            (label_x, ema_label_x), label_gt = next(train_label_iter)

        batch_size = unlabel_x.size(0)
        label_batch_size = label_x.size(0)

        label_x = Variable(label_x).cuda()
        ema_label_x = Variable(ema_label_x).cuda()
        unlabel_x = Variable(unlabel_x).cuda()
        ema_unlabel_x = Variable(ema_unlabel_x).cuda()
        label_gt = Variable(label_gt.type(LongTensor), requires_grad=False).cuda()

        _, label_logits = model(label_x)
        _, unlabel_logits = model(unlabel_x)
        _, ema_label_logits = ema_model(ema_label_x)
        _, ema_unlabel_logits = ema_model(ema_unlabel_x)

        ce_loss = cross_entropy_loss(label_logits, label_gt)

        consistency_loss = (softmax_mse_loss(label_logits, ema_label_logits) * label_batch_size +
                            softmax_mse_loss(unlabel_logits, ema_unlabel_logits) * batch_size)\
                           /(label_batch_size + batch_size)

        loss = ce_loss + consistency_loss * data_params.mt_consistency_weight * \
                         sigmoid_rampup(epoch, data_params.mt_rampup_epochs)

        # Compute EMA loss
        ema_ce_loss = cross_entropy_loss(ema_label_logits, label_gt)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update EMA model param:
        update_ema_variables(model, ema_model, ema_decay=data_params.mt_ema_decay)

        losses.update(ce_loss.item(), label_batch_size)
        # accuracy
        prec1, prec5 = accuracy(label_logits.data, label_gt.data, topk=(1, 5))
        top1.update(prec1[0], label_batch_size)
        top5.update(prec5[0], label_batch_size)

        ema_losses.update(ema_ce_loss.item(), label_batch_size)
        # accuracy
        ema_prec1, ema_prec5 = accuracy(ema_label_logits.data, label_gt.data, topk=(1, 5))
        ema_top1.update(ema_prec1[0], label_batch_size)
        ema_top5.update(ema_prec5[0], label_batch_size)

        consistency_losses.update(consistency_loss.item(), batch_size + label_batch_size)

    logging.info(('Training Epoch: {0}, '
              'lr: {lr:.5f}, '
              'Prec@1 {top1.avg:.4f}, '
              'Prec@5 {top5.avg:.4f}, '
              'Loss {losses.avg:.4f}, '
              'Consistency Loss {consistency_losses.avg:.4f}, '
              'EMA_Prec@1 {ema_top1.avg:.4f}, '
              'EMA_Prec@5 {ema_top5.avg:.4f}, '
              'EMA_Loss {ema_losses.avg:.4f}, '
              .format(epoch, lr=optimizer.param_groups[-1]['lr'], top1=top1, top5=top5, losses=losses,
                      consistency_losses=consistency_losses, ema_losses=ema_losses, ema_top1=ema_top1,
                      ema_top5=ema_top5)))

    return top1.avg


if __name__ == '__main__':
    main()


