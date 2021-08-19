import torch.nn.functional as F
import torch
from .triplet_loss import TripletLoss
from .cross_entropy_loss import CrossEntropyLoss


def make_loss(cfg):
    sampler = cfg.DATALOADER.SAMPLER
    triplet = TripletLoss(cfg.SOLVER.MARGIN)  # triplet loss
    cross_entropy = CrossEntropyLoss(num_classes=cfg.SOLVER.CLASSNUM,epsilon=cfg.SOLVER.SMOOTH)

    if sampler == 'softmax':
        def loss_func(score, feat, target):
            return F.cross_entropy(score, target)
    elif cfg.DATALOADER.SAMPLER == 'triplet':
        def loss_func(score, feat, target):
            return triplet(feat, target)[0]
    elif cfg.DATALOADER.SAMPLER == 'softmax_triplet':
        if cfg.MODEL.CAL:
            def loss_func(score,score_hat, feat, target):
                loss_id = cross_entropy(score, target)  + triplet(feat, target)[0] + cross_entropy(score_hat, target)
                return loss_id
        else:
            def loss_func(score, feat, target):
                loss_id = cross_entropy(score, target)  + triplet(feat, target)[0]
                return loss_id
    else:
        print('expected sampler should be softmax, triplet or softmax_triplet, '
              'but got {}'.format(cfg.DATALOADER.SAMPLER))
    return loss_func
