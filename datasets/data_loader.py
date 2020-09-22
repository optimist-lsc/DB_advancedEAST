

import imgaug
import numpy as np

import torch
from torch.utils.data import Sampler


def default_worker_init_fn(worker_id):
    np.random.seed(worker_id)
    imgaug.seed(worker_id)


class DataLoader(torch.utils.data.DataLoader):

    def __init__(self, dataset,batch_size,num_workers,is_train=False,collect_fn=None):
        if collect_fn is None:
            self.collect_fn = torch.utils.data.dataloader.default_collate
        else:
            self.collect_fn = collect_fn
        self.dataset = dataset
        self.is_train = is_train
        self.batch_size = batch_size
        self.shuffle = self.is_train
        self.num_workers = num_workers
        self.drop_last = True


        torch.utils.data.DataLoader.__init__(
            self, self.dataset,
            batch_size=self.batch_size, num_workers=self.num_workers,
            drop_last=self.drop_last, shuffle=self.shuffle,
            pin_memory=True, collate_fn=self.collect_fn,
            worker_init_fn=default_worker_init_fn)
        self.collect_fn = str(self.collect_fn)



