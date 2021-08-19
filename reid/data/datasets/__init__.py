from .dukemtmcreid import DukeMTMCreID
from .market1501 import Market1501
from .msmt17_v1 import MSMT17_V1
from .dataset_loader import ImageDataset

__factory = {
    'market1501': Market1501,
    'dukemtmc': DukeMTMCreID,
    'msmt17': MSMT17_V1
}


def get_names():
    return __factory.keys()


def init_dataset(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown datasets: {}".format(name))
    return __factory[name](*args, **kwargs)
