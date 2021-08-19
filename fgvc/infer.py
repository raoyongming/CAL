import os
import config_infer as config

os.environ['CUDA_VISIBLE_DEVICES'] = config.GPU
import time
import logging
import warnings
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models import WSDAN_CAL
from datasets import get_trainval_datasets
from utils import CenterLoss, AverageMeter, TopKAccuracyMetric, ModelCheckpoint, batch_augment

# GPU settings
assert torch.cuda.is_available()
os.environ['CUDA_VISIBLE_DEVICES'] = config.GPU
device = torch.device("cuda")
torch.backends.cudnn.benchmark = True

# General loss functions
cross_entropy_loss = nn.CrossEntropyLoss()
center_loss = CenterLoss()

# loss and metric
loss_container = AverageMeter(name='loss')
raw_metric = TopKAccuracyMetric(topk=(1, 5))
crop_metric = TopKAccuracyMetric(topk=(1, 5))
drop_metric = TopKAccuracyMetric(topk=(1, 5))

best_acc = 0.0
from torchvision import transforms
ToPILImage = transforms.ToPILImage()
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

def generate_heatmap(attention_maps):
    heat_attention_maps = []
    heat_attention_maps.append(attention_maps[:, 0, ...])  # R
    heat_attention_maps.append(attention_maps[:, 0, ...] * (attention_maps[:, 0, ...] < 0.5).float() + \
                               (1. - attention_maps[:, 0, ...]) * (attention_maps[:, 0, ...] >= 0.5).float())  # G
    heat_attention_maps.append(1. - attention_maps[:, 0, ...])  # B
    return torch.stack(heat_attention_maps, dim=1)

def main():
    # Load dataset
    ##################################
    train_dataset, validate_dataset = get_trainval_datasets(config.tag, config.image_size)

    validate_loader = DataLoader(validate_dataset, batch_size=config.batch_size, shuffle=False,
                                               num_workers=config.workers, pin_memory=True)
    num_classes = 200

    ##################################
    # Initialize model
    ##################################
    logs = {}
    start_epoch = 0
    net = WSDAN_CAL(num_classes=num_classes, M=config.num_attentions, net=config.net, pretrained=True)

    checkpoint = torch.load(config.ckpt)
    state_dict = checkpoint['state_dict']
    net.load_state_dict(state_dict)
    print('Network loaded from {}'.format(config.ckpt))

    net.to(device)
    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)

    if config.visual_path is not None:
        visualize(data_loader=validate_loader, net=net)
    test(data_loader=validate_loader, net=net)

def visualize(**kwargs):
    data_loader = kwargs['data_loader']
    net = kwargs['net']

    # metrics initialization
    loss_container.reset()
    raw_metric.reset()
    drop_metric.reset()

    # begin validation
    start_time = time.time()
    net.eval()

    savepath = config.visual_path  # './vis_counterfactual/'

    if not os.path.isdir(savepath):
        os.mkdir(savepath)

    with torch.no_grad():
        for i, (X, y) in enumerate(data_loader):
            X = X.to(device)

            attention_maps = net.visualize(X)
            attention_maps = torch.max(attention_maps, dim=1, keepdim=True)[0]
            attention_maps = F.upsample_bilinear(attention_maps, size=(X.size(2), X.size(3)))
            attention_maps = torch.sqrt(attention_maps.cpu() / attention_maps.max().item())

            # get heat attention maps
            heat_attention_maps = generate_heatmap(attention_maps)

            # raw_image, heat_attention, raw_attention
            raw_image = X.cpu() * STD + MEAN
            heat_attention_image = raw_image * 0.5 + heat_attention_maps * 0.5
            raw_attention_image = raw_image * attention_maps

            for batch_idx in range(X.size(0)):
                rimg = ToPILImage(raw_image[batch_idx])
                # raimg = ToPILImage(raw_attention_image[batch_idx])
                haimg = ToPILImage(heat_attention_image[batch_idx])
                rimg.save(os.path.join(savepath, '%03d_raw.jpg' % (i * config.batch_size + batch_idx)))
                # raimg.save(os.path.join(savepath, '%03d_raw_atten.jpg' % (i * config.batch_size + batch_idx)))
                haimg.save(os.path.join(savepath, '%03d_heat_atten.jpg' % (i * config.batch_size + batch_idx)))

            print('iter %d / %d done!' % (i, len(data_loader)))


def test(**kwargs):
    # Retrieve training configuration
    global best_acc
    data_loader = kwargs['data_loader']
    net = kwargs['net']

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
            X = X.to(device)
            y = y.to(device)

            X_m = torch.flip(X, [3])

            ##################################
            # Raw Image
            ##################################
            y_pred_raw, y_pred_aux_raw, _, attention_map = net(X)
            y_pred_raw_m, y_pred_aux_raw_m, _, attention_map_m = net(X_m)

            ##################################
            # Object Localization and Refinement
            ##################################

            crop_images = batch_augment(X, attention_map, mode='crop', theta=0.3, padding_ratio=0.1)
            y_pred_crop, y_pred_aux_crop, _, _ = net(crop_images)

            crop_images2 = batch_augment(X, attention_map, mode='crop', theta=0.2, padding_ratio=0.1)
            y_pred_crop2, y_pred_aux_crop2, _, _ = net(crop_images2)

            crop_images3 = batch_augment(X, attention_map, mode='crop', theta=0.1, padding_ratio=0.05)
            y_pred_crop3, y_pred_aux_crop3, _, _ = net(crop_images3)

            crop_images_m = batch_augment(X_m, attention_map_m, mode='crop', theta=0.3, padding_ratio=0.1)
            y_pred_crop_m, y_pred_aux_crop_m, _, _ = net(crop_images_m)

            crop_images_m2 = batch_augment(X_m, attention_map_m, mode='crop', theta=0.2, padding_ratio=0.1)
            y_pred_crop_m2, y_pred_aux_crop_m2, _, _ = net(crop_images_m2)

            crop_images_m3 = batch_augment(X_m, attention_map_m, mode='crop', theta=0.1, padding_ratio=0.05)
            y_pred_crop_m3, y_pred_aux_crop_m3, _, _ = net(crop_images_m3)

            y_pred = (y_pred_raw + y_pred_crop + y_pred_crop2 + y_pred_crop3) / 4.
            y_pred_m = (y_pred_raw_m + y_pred_crop_m + y_pred_crop_m2 + y_pred_crop_m3) / 4.
            y_pred = (y_pred + y_pred_m) / 2.

            y_pred_aux = (y_pred_aux_raw + y_pred_aux_crop + y_pred_aux_crop2 + y_pred_aux_crop3) / 4.
            y_pred_aux_m = (y_pred_aux_raw_m + y_pred_aux_crop_m + y_pred_aux_crop_m2 + y_pred_aux_crop_m3) / 4.
            y_pred_aux = (y_pred_aux + y_pred_aux_m) / 2.

            # loss
            batch_loss = cross_entropy_loss(y_pred, y)
            epoch_loss = loss_container(batch_loss.item())

            # metrics: top-1,5 error
            epoch_acc = raw_metric(y_pred, y)
            aux_acc = drop_metric(y_pred_aux, y)

            batch_info = 'Val Loss {:.4f}, Val Acc ({:.2f}, {:.2f}), Val Aux Acc ({:.2f}, {:.2f}), Best {:.2f}'.format(
                epoch_loss, epoch_acc[0], epoch_acc[1], aux_acc[0], aux_acc[1], best_acc)
            if i % 5 == 0:
                print('%d/%d:' % (i, len(data_loader)), batch_info)

    batch_info = 'Val Loss {:.4f}, Val Acc ({:.2f}, {:.2f}), Val Aux Acc ({:.2f}, {:.2f}), Best {:.2f}'.format(epoch_loss, epoch_acc[0], epoch_acc[1], aux_acc[0], aux_acc[1], best_acc)
    print(batch_info)


if __name__ == '__main__':
    main()
