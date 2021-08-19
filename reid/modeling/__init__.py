from .baseline import Baseline
import torch.nn as nn


def build_model(cfg, num_classes):
    if cfg.MODEL.NAME == 'resnet50':
        model = Baseline(num_classes, cfg.MODEL.LAST_STRIDE, cfg.MODEL.PRETRAIN_PATH, cfg.MODEL.CAL)
    return model
