""" Stanford Cars (Car) Dataset """
import os
import pdb
from PIL import Image
from scipy.io import loadmat
from torch.utils.data import Dataset
from utils import get_transform

DATAPATH = './stanford_cars/'


class CarDataset(Dataset):
    """
    # Description:
        Dataset for retrieving Stanford Cars images and labels

    # Member Functions:
        __init__(self, phase, resize):  initializes a dataset
            phase:                      a string in ['train', 'val', 'test']
            resize:                     output shape/size of an image

        __getitem__(self, item):        returns an image
            item:                       the idex of image in the whole dataset

        __len__(self):                  returns the length of dataset
    """

    def __init__(self, phase='train', resize=500):
        assert phase in ['train', 'val', 'test']
        self.phase = phase
        self.resize = resize
        self.num_classes = 196

        self.images = []
        self.labels = []

        list_path = os.path.join(DATAPATH, 'cars_annos.mat')

        list_mat = loadmat(list_path)
        num_inst = len(list_mat['annotations']['relative_im_path'][0])
        for i in range(num_inst):
            if phase == 'train' and list_mat['annotations']['test'][0][i].item() == 0:
                path = list_mat['annotations']['relative_im_path'][0][i].item()
                label = list_mat['annotations']['class'][0][i].item()
                self.images.append(path)
                self.labels.append(label)
            elif phase != 'train' and list_mat['annotations']['test'][0][i].item() == 1:
                path = list_mat['annotations']['relative_im_path'][0][i].item()
                label = list_mat['annotations']['class'][0][i].item()
                self.images.append(path)
                self.labels.append(label)

        print('Car Dataset with {} instances for {} phase'.format(len(self.images), self.phase))

        # transform
        self.transform = get_transform(self.resize, self.phase)

    def __getitem__(self, item):
        # image
        image = Image.open(os.path.join(DATAPATH, self.images[item])).convert('RGB')  # (C, H, W)
        image = self.transform(image)

        # return image and label
        return image, self.labels[item] - 1  # count begin from zero

    def __len__(self):
        return len(self.images)


if __name__ == '__main__':
    ds = CarDataset('val')
    # print(len(ds))
    for i in range(0, 100):
        image, label = ds[i]
        # print(image.shape, label)
