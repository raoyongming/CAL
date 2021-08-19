import logging
import torch
import numpy as np
from ignite.engine import Engine

from utils.reid_metric import R1_mAP


def create_supervised_evaluator(model, metrics,
                                device=None):
    if device:
        model.to(device)

    def fliplr(img):
        inv_idx = torch.arange(img.size(3)-1,-1,-1).long().cuda()  # N x C x H x W
        img_flip = img.index_select(3,inv_idx)
        return img_flip

    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            data, pids, camids = batch
            data = data.cuda()
            feat = model(data)
            data_f = fliplr(data) 
            feat_f  = model(data_f)
            feat = feat + feat_f
            return feat, pids, camids

    engine = Engine(_inference)
    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine


def inference(
        cfg,
        model,
        val_loader,
        num_query
):
    device = cfg.MODEL.DEVICE

    logger = logging.getLogger("reid_baseline.inference")
    logger.info("Start inferencing")
    evaluator = create_supervised_evaluator(model, metrics={'r1_mAP': R1_mAP(num_query,re_rank=cfg.TEST.RE_RANK)},
                                            device=device)

    evaluator.run(val_loader)
    cmc, mAP = evaluator.state.metrics['r1_mAP']
    logger.info('Validation Results')
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))

