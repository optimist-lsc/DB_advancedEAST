
import os
import torch.utils.data as data
import cv2
import numpy as np
import math

from .augment_data import AugmentData
from .random_crop_data import RandomCropData
from .make_icdar_data import MakeICDARData
from .make_label import MakeLabel
from .normalize_image import NormalizeImage
from .filter_keys import FilterKeys


class ImageDataset(data.Dataset):
    r'''Dataset reading from images.
    Args:
        Processes: A series of Callable object, which accept as parameter and return the data dict,
            typically inherrited the `DataProcess`(data/processes/data_process.py) class.
    '''
    def __init__(self, img_dir=None,gt_dir=None,input_size=640):
        self.img_dir = img_dir
        self.gt_dir = gt_dir
        self.input_size = input_size
        if 'train' in self.img_dir:
            self.is_training = True
        else:
            self.is_training = False
        self.image_paths = []
        self.gt_paths = []
        self.get_all_samples()
        self._init_pre_processes()

    def get_all_samples(self):
        for imgName in os.listdir(self.img_dir):
            img_path = os.path.join(self.img_dir, imgName)
            gt_paths = os.path.join(self.gt_dir, imgName[:-4] + '.txt')
            # gt_paths = os.path.join(self.gt_dir, 'gt'+imgName[3:-4] + '.txt')
            self.image_paths.append(img_path)
            self.gt_paths.append(gt_paths)
        self.num_samples = len(self.image_paths)
        self.targets = self.load_ann()
        if self.is_training:
            assert len(self.image_paths) == len(self.targets)

    def _init_pre_processes(self):
        self.processes = []
        if self.is_training:
            self.processes.append(AugmentData())
            self.processes.append(RandomCropData(size=[self.input_size, self.input_size],max_tries=10))
            self.processes.append(MakeICDARData())
            self.processes.append(MakeLabel())
            self.processes.append(NormalizeImage())
            self.processes.append(FilterKeys())
        else:
            self.processes.append(AugmentData((800, 800),only_resize=True,keep_ratio=True))
            self.processes.append(MakeICDARData())
            self.processes.append(NormalizeImage())

    def load_ann(self):
        res = []
        for gt in self.gt_paths:
            lines = []
            reader = open(gt, 'r').readlines()
            for line in reader:
                item = {}
                parts = line.strip().split(',')
                label = parts[-1]
                if label == '1':
                    label = '###'
                line = [i.strip('\ufeff').strip('\xef\xbb\xbf') for i in parts]

                poly = np.array(list(map(float, line[:8]))).reshape((-1, 2)).tolist()

                item['poly'] = poly
                item['text'] = label
                lines.append(item)
            res.append(lines)
        return res

    def __getitem__(self, index, retry=0):
        if index >= self.num_samples:
            index = index % self.num_samples
        data = {}
        image_path = self.image_paths[index]
        img = cv2.imread(image_path, cv2.IMREAD_COLOR).astype('float32')
        if self.is_training:
            data['filename'] = image_path
            data['data_id'] = image_path
        else:
            data['filename'] = image_path.split('/')[-1]
            data['data_id'] = image_path.split('/')[-1]
        data['image'] = img
        target = self.targets[index]
        data['lines'] = target
        if self.processes is not None:
            for data_process in self.processes:
                data = data_process(data)


        return data

    def __len__(self):
        return len(self.image_paths)
