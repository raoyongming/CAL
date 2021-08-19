import glob
import re
import urllib
import zipfile
import os.path as osp

from utils.iotools import mkdir_if_missing
from .bases import BaseImageDataset


class MSMT17_V1(BaseImageDataset):
    dataset_dir = 'MSMT17_V1'

    def __init__(self, root='your_dataset_path', verbose=True, **kwargs):
        super(MSMT17_V1, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.gallery_dir = osp.join(self.dataset_dir, 'test')
        self.list_train_path = osp.join(self.dataset_dir, 'list_train.txt')
        self.list_val_path = osp.join(self.dataset_dir, 'list_val.txt')
        self.list_query_path = osp.join(self.dataset_dir, 'list_query.txt')
        self.list_gallery_path = osp.join(self.dataset_dir, 'list_gallery.txt')
        self._download_data()
        self._check_before_run()

        train = self._process_dir(self.train_dir, self.list_train_path, relabel=True)
        query = self._process_dir(self.gallery_dir, self.list_query_path, relabel=False)
        gallery = self._process_dir(self.gallery_dir, self.list_gallery_path, relabel=False)

        if verbose:
            print("=> MSMT17_V1 loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

    def _download_data(self):
        if osp.exists(self.dataset_dir):
            print("This dataset has been downloaded.")
            return

        print("Creating directory {}".format(self.dataset_dir))
        mkdir_if_missing(self.dataset_dir)
        fpath = osp.join(self.dataset_dir, osp.basename(self.dataset_url))

        print("Downloading MSMT17_V1 dataset")
        urllib.urlretrieve(self.dataset_url, fpath)

        print("Extracting files")
        zip_ref = zipfile.ZipFile(fpath, 'r')
        zip_ref.extractall(self.dataset_dir)
        zip_ref.close()

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.list_query_path):
            raise RuntimeError("'{}' is not available".format(self.list_query_path))
        if not osp.exists(self.list_gallery_path):
            raise RuntimeError("'{}' is not available".format(self.list_gallery_path))

    def _process_dir(self, dir_path, txt_path, relabel=False):
        with open(txt_path, 'r') as f:
            img_paths = f.read()
        img_paths = img_paths.split()
        pid_container = set()

        for pid in img_paths[1::2]:
            pid = int(pid)
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        dataset = []
        for img_path in img_paths[::2]:
            pid = int(img_path.split('/')[0])
            camid = int(img_path.split('_')[2])
            assert 1 <= camid <= 15
            camid -= 1  # index starts from 0
            if relabel: pid = pid2label[pid]
            img_path = osp.join(dir_path,img_path)
            dataset.append((img_path, pid, camid))

        return dataset
