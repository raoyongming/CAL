import numpy as np
import torch
from ignite.metrics import Metric
from data.datasets.eval_reid import eval_func
from utils.re_ranking import re_ranking


def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist

def euclidean_dist_cpu(x, y): # evaluation on MSMT17
    m, n = x.shape[0], y.shape[0]
    xx = np.power(x, 2).sum(1)
    xx = np.reshape(xx, [xx.shape[0], 1])
    xx = xx.repeat(n, axis=1)
    yy = np.power(y, 2).sum(1)
    yy = np.reshape(yy, [yy.shape[0], 1])
    yy = yy.repeat(m, axis=1).T
    dist = xx + yy
    dist -= 2 * np.dot(x, y.T)
    dist = np.sqrt(np.clip(dist, 1e-12, dist.max()))
    return dist

def cos_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    xx = x/x.norm(dim=1)[:,None]
    yy = y/y.norm(dim=1)[:,None]
    dist = torch.mm(xx,yy.t())
    # dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return 1-dist

class R1_mAP(Metric):
    def __init__(self, num_query, max_rank=50, re_rank = False):
        super(R1_mAP, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.re_rank = re_rank
        self.count = 0



    def reset(self):
        self.feats = []
        self.pids = []
        self.camids = []

    def update(self, output):
        feat, pid, camid = output
        self.feats.append(feat)
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))

    def compute(self):
        feats = torch.cat(self.feats, dim=0)
        fnorm = torch.norm(feats,p=2,dim=1,keepdim=True)
        feats = feats.div(fnorm.expand_as(feats))
        # query
        qf = feats[:self.num_query]
        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])
        # gallery
        gf = feats[self.num_query:]
        g_pids = np.asarray(self.pids[self.num_query:])
        g_camids = np.asarray(self.camids[self.num_query:])


        qf = qf.cpu().numpy()
        gf = gf.cpu().numpy()
        distmat_q_g = euclidean_dist_cpu(qf,gf)

        if self.re_rank:
        #distmat_cos = cos_dist(qf,gf)
            distmat_q_q = euclidean_dist(qf,qf)
            distmat_g_g = euclidean_dist(gf,gf)
            distmat_q_q = distmat_q_q.cpu().numpy()
            distmat_g_g = distmat_g_g.cpu().numpy()
            distmat = re_ranking(distmat_q_g,distmat_q_q,distmat_g_g )
        else:
            distmat = distmat_q_g
        cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)

        return cmc, mAP
