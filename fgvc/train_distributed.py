import os
import config_distributed as config

import time
import logging
import warnings
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import random
from models import WSDAN_CAL
from utils import CenterLoss, AverageMeter, TopKAccuracyMetric, ModelCheckpoint, batch_augment
from datasets import get_trainval_datasets
import math
from apex import amp
import apex
from apex.parallel import DistributedDataParallel as DDP

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--local_rank', type=int, default=0)
args = parser.parse_args()


# General loss functions
cross_entropy_loss = nn.CrossEntropyLoss()
center_loss = CenterLoss()

# loss and metric
loss_container = AverageMeter(name='loss')
top1_container = AverageMeter(name='top1')
top5_container = AverageMeter(name='top5')

raw_metric = TopKAccuracyMetric(topk=(1, 5))
crop_metric = TopKAccuracyMetric(topk=(1, 5))
drop_metric = TopKAccuracyMetric(topk=(1, 5))

best_acc = 0.0

def main():
    torch.cuda.set_device(args.local_rank)
    torch.backends.cudnn.benchmark = True

    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    args.world_size = torch.distributed.get_world_size()

    ##################################
    # Logging setting
    ##################################
    if args.local_rank == 0:
        if not os.path.exists(config.save_dir):
            os.makedirs(config.save_dir)

        logging.basicConfig(
            filename=os.path.join(config.save_dir, config.log_name),
            filemode='w',
            format='%(asctime)s: %(levelname)s: [%(filename)s:%(lineno)d]: %(message)s',
            level=logging.INFO)
        warnings.filterwarnings("ignore")

    train_dataset, validate_dataset = get_trainval_datasets(config.tag, config.image_size)

    num_classes = train_dataset.num_classes

    ##################################
    # Initialize model
    ##################################
    logs = {}
    start_epoch = 0
    net = WSDAN_CAL(num_classes=num_classes, M=config.num_attentions, net=config.net, pretrained=True)

    # feature_center: size of (#classes, #attention_maps * #channel_features)
    feature_center = torch.zeros(num_classes, config.num_attentions * net.num_features).cuda()

    if config.ckpt and os.path.isfile(config.ckpt):
        # Load ckpt and get state_dict
        checkpoint = torch.load(config.ckpt)

        # Get epoch and some logs
        logs = checkpoint['logs']
        start_epoch = int(logs['epoch']) # start from the beginning

        # Load weights
        state_dict = checkpoint['state_dict']
        net.load_state_dict(state_dict)
        if args.local_rank == 0:
            logging.info('Network loaded from {}'.format(config.ckpt))
            print('Network loaded from {} @ {} epoch'.format(config.ckpt, start_epoch))

        # load feature center
        if 'feature_center' in checkpoint:
            feature_center = checkpoint['feature_center'].cuda()
            if args.local_rank == 0:
                logging.info('feature_center loaded from {}'.format(config.ckpt))

    if args.local_rank == 0:
        logging.info('Network weights save to {}'.format(config.save_dir))

    ##################################
    # Use cuda
    ##################################
    print("using apex synced BN")
    net = apex.parallel.convert_syncbn_model(net)
    net.cuda()

    learning_rate = config.learning_rate
    print('begin with', learning_rate, 'learning rate')
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-5)

    net, optimizer = amp.initialize(net, optimizer, opt_level='O0')
    net = DDP(net, delay_allreduce=True)


    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    val_sampler = torch.utils.data.distributed.DistributedSampler(validate_dataset)

    train_loader, validate_loader = DataLoader(train_dataset, batch_size=config.batch_size, sampler=train_sampler,
                                               num_workers=config.workers, pin_memory=True, drop_last=True), \
                                    DataLoader(validate_dataset, batch_size=config.batch_size * 4, sampler=val_sampler,
                                               num_workers=config.workers, pin_memory=True, drop_last=True)


    if args.local_rank == 0:
        callback_monitor = 'val_{}'.format(raw_metric.name)
        callback = ModelCheckpoint(savepath=os.path.join(config.save_dir, config.model_name),
                                   monitor=callback_monitor,
                                   mode='max')
        if callback_monitor in logs:
            callback.set_best_score(logs[callback_monitor])
        else:
            callback.reset()
        logging.info('Start training: Total epochs: {}, Batch size: {}, Training size: {}, Validation size: {}'.
                     format(config.epochs, config.batch_size, len(train_dataset), len(validate_dataset)))
        logging.info('')

    for epoch in range(start_epoch, config.epochs):
        if args.local_rank == 0:
            callback.on_epoch_begin()
            logs['epoch'] = epoch + 1
            logs['lr'] = optimizer.param_groups[0]['lr']
            print('current lr =', optimizer.param_groups[0]['lr'])

            logging.info('Epoch {:03d}, Learning Rate {:g}'.format(epoch + 1, optimizer.param_groups[0]['lr']))

        if args.local_rank == 0:
            pbar = tqdm(total=len(train_loader), unit=' batches')
            pbar.set_description('Epoch {}/{}'.format(epoch + 1, config.epochs))
        else:
            pbar = None

        train_sampler.set_epoch(epoch)
        train(epoch=epoch,
              logs=logs,
              data_loader=train_loader,
              net=net,
              feature_center=feature_center,
              optimizer=optimizer,
              pbar=pbar)

        validate(logs=logs,
                 data_loader=validate_loader,
                 net=net,
                 pbar=pbar,
                 epoch=epoch)

        torch.cuda.synchronize()
        if args.local_rank == 0:
            callback.on_epoch_end(logs, net, feature_center=feature_center)
            pbar.close()

def adjust_learning(optimizer, epoch, iter):
    """Decay the learning rate based on schedule"""
    base_lr = config.learning_rate
    base_rate = 0.9
    base_duration = 2.0
    lr = base_lr * pow(base_rate, (epoch + iter) / base_duration)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(**kwargs):
    # Retrieve training configuration
    epoch = kwargs['epoch']
    logs = kwargs['logs']
    data_loader = kwargs['data_loader']
    net = kwargs['net']
    feature_center = kwargs['feature_center']
    optimizer = kwargs['optimizer']
    pbar = kwargs['pbar']

    # metrics initialization
    loss_container.reset()
    raw_metric.reset()
    crop_metric.reset()
    drop_metric.reset()

    # begin training
    start_time = time.time()
    net.train()
    batch_len = len(data_loader)
    for i, (X, y) in enumerate(data_loader):
        float_iter = float(i) / batch_len
        adjust_learning(optimizer, epoch, float_iter)
        now_lr = optimizer.param_groups[0]['lr']

        optimizer.zero_grad()

        # obtain data for training
        X = X.cuda()
        y = y.cuda()

        y_pred_raw, y_pred_aux, feature_matrix, attention_map = net(X)

        # Update Feature Center
        feature_center_batch = F.normalize(feature_center[y], dim=-1)
        feature_center[y] += config.beta * (feature_matrix.detach() - feature_center_batch)

        ##################################
        # Attention Cropping
        ##################################
        with torch.no_grad():
            crop_images = batch_augment(X, attention_map[:, :1, :, :], mode='crop', theta=(0.4, 0.6), padding_ratio=0.1)
            drop_images = batch_augment(X, attention_map[:, 1:, :, :], mode='drop', theta=(0.2, 0.5))
        aug_images = torch.cat([crop_images, drop_images], dim=0)
        y_aug = torch.cat([y, y], dim=0)

        # crop images forward
        y_pred_aug, y_pred_aux_aug, _, _ = net(aug_images)

        y_pred_aux = torch.cat([y_pred_aux, y_pred_aux_aug], dim=0)
        y_aux = torch.cat([y, y_aug], dim=0)

        # loss
        batch_loss = cross_entropy_loss(y_pred_raw, y) / 3. + \
                     cross_entropy_loss(y_pred_aux, y_aux) * 3. / 3. + \
                     cross_entropy_loss(y_pred_aug, y_aug) * 2. / 3. + \
                     center_loss(feature_matrix, feature_center_batch)

        # backward
        batch_loss.backward()
        optimizer.step()

        # metrics: loss and top-1,5 error
        with torch.no_grad():
            epoch_loss = loss_container(batch_loss.item())
            epoch_raw_acc = raw_metric(y_pred_raw, y)
            epoch_crop_acc = crop_metric(y_pred_aug, y_aug)
            epoch_drop_acc = drop_metric(y_pred_aux, y_aux)

        # end of this batch
        batch_info = 'Loss {:.4f}, Raw Acc ({:.2f}, {:.2f}), Aug Acc ({:.2f}, {:.2f}), Aux Acc ({:.2f}, {:.2f}), lr {:.5f}'.format(
            epoch_loss, epoch_raw_acc[0], epoch_raw_acc[1],
            epoch_crop_acc[0], epoch_crop_acc[1], epoch_drop_acc[0], epoch_drop_acc[1], now_lr)

        if args.local_rank == 0:
            pbar.update()
            pbar.set_postfix_str(batch_info)

    # end of this epoch
    logs['train_{}'.format(loss_container.name)] = epoch_loss
    logs['train_raw_{}'.format(raw_metric.name)] = epoch_raw_acc
    logs['train_crop_{}'.format(crop_metric.name)] = epoch_crop_acc
    logs['train_drop_{}'.format(drop_metric.name)] = epoch_drop_acc
    logs['train_info'] = batch_info
    end_time = time.time()

    # write log for this epoch
    logging.info('Train: {}, Time {:3.2f}'.format(batch_info, end_time - start_time))


def validate(**kwargs):
    # Retrieve training configuration
    global best_acc
    epoch = kwargs['epoch']
    logs = kwargs['logs']
    data_loader = kwargs['data_loader']
    net = kwargs['net']
    pbar = kwargs['pbar']

    # metrics initialization
    loss_container.reset()
    raw_metric.reset()
    drop_metric.reset()

    # begin validation
    start_time = time.time()
    net.eval()
    with torch.no_grad():
        for i, (X, y) in enumerate(data_loader):
            # obtain data
            X = X.cuda()
            y = y.cuda()

            ##################################
            # Raw Image
            ##################################
            y_pred_raw, y_pred_aux, _, attention_map = net(X)

            crop_images3 = batch_augment(X, attention_map, mode='crop', theta=0.1, padding_ratio=0.05)
            y_pred_crop3, y_pred_aux_crop3, _, _ = net(crop_images3)

            ##################################
            # Final prediction
            ##################################
            y_pred = (y_pred_raw + y_pred_crop3) / 2.
            y_pred_aux = (y_pred_aux + y_pred_aux_crop3) / 2.

            # loss
            batch_loss = cross_entropy_loss(y_pred, y)
            batch_loss = reduce_tensor(batch_loss.data)
            epoch_loss = loss_container(batch_loss.item())

            y_pred = gather_tensor(y_pred)
            y_pred_aux = gather_tensor(y_pred_aux)
            y = gather_tensor(y)

            # metrics: top-1,5 error
            epoch_acc = raw_metric(y_pred, y)
            aux_acc = drop_metric(y_pred_aux, y)

    # end of validation
    logs['val_{}'.format(loss_container.name)] = epoch_loss
    logs['val_{}'.format(raw_metric.name)] = epoch_acc
    end_time = time.time()

    batch_info = 'Val Loss {:.4f}, Val Acc ({:.2f}, {:.2f})'.format(epoch_loss, epoch_acc[0], epoch_acc[1])

    if args.local_rank == 0:
        pbar.set_postfix_str('{}, {}'.format(logs['train_info'], batch_info))

        if epoch_acc[0] > best_acc:
            best_acc = epoch_acc[0]
            save_model(net, logs, 'model_bestacc.pth')

        if aux_acc[0] > best_acc:
            best_acc = aux_acc[0]
            save_model(net, logs, 'model_bestacc.pth')

        if epoch % 10 == 0:
            save_model(net, logs, 'model_epoch%d.pth' % epoch)


        batch_info = 'Val Loss {:.4f}, Val Acc ({:.2f}, {:.2f}), Val Aux Acc ({:.2f}, {:.2f}), Best {:.2f}'.format(
            epoch_loss, epoch_acc[0], epoch_acc[1], aux_acc[0], aux_acc[1], best_acc)
        print(batch_info)

        # write log for this epoch
        logging.info('Valid: {}, Time {:3.2f}'.format(batch_info, end_time - start_time))
        logging.info('')

def save_model(net, logs, ckpt_name):
    torch.save({'logs': logs, 'state_dict': net.module.state_dict()}, config.save_dir + 'model_bestacc.pth')

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

def reduce_tensor(tensor):
    rt = tensor.clone()
    torch.distributed.all_reduce(rt, op=torch.distributed.reduce_op.SUM)
    rt /= args.world_size
    return rt

def gather_tensor(tensor):
    rt = tensor.clone()
    gather_t = [torch.ones_like(rt) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(gather_t, rt)
    gather_t = torch.cat(gather_t, dim=0)
    return gather_t

if __name__ == '__main__':
    main()

