import torch
from torch.utils import data
from torch import nn
from datasets.image_dataset import ImageDataset
from model.DB_model import EAST

import numpy as np
from tqdm import tqdm

import cfg
from utils.nms_v2 import nms
from datasets.make_icdar_data import ICDARCollectFN
from utils.preprocess import reorder_vertexes
from concern.quad_measurer import QuadMeasurer
from utils.log import Logger

logger = Logger('saved_model')

valset = ImageDataset(cfg.val_image_dir,cfg.val_label_dir)

val_loader = data.DataLoader(valset, batch_size=1, \
							   shuffle=False, num_workers=0, pin_memory=True, drop_last=True,collate_fn=ICDARCollectFN())

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = EAST()


model = nn.DataParallel(model)
model.to(device)
model.load_state_dict(torch.load('./model/resnest_east_weights_3T800_600.h5'))

model.eval()
raw_metrics = []
for i, (batch) in tqdm(enumerate(val_loader)):
	img = batch['image'].to(device)
	with torch.no_grad():
		east_detect = model(img)
	# east_detect[:, :3, :, :] = torch.sigmoid(east_detect[:, :3, :, :])
	y = torch.squeeze(east_detect, axis=0)  # [1,7,w,h]转[7,w,h] 删除单维度条目
	y = y.cpu().numpy()
	height,width = y.shape[1:]
	# 前三个通道需要sigmod，转为概率
	cond = np.greater_equal(y[0, :, :], cfg.pixel_threshold)  # 筛选出大于阈值的值,大于阈值为true，小于阈值为false
	activation_pixels = np.where(cond)  # 大于零的坐标点

	quad_scores, quad_after_nms = nms(y, activation_pixels)


	boxes_batch = []

	boxes = []

	for score, geo, index in zip(quad_scores, quad_after_nms,
							 range(len(quad_scores))):
		if np.amin(score) > 0:
			geo = reorder_vertexes(geo)[[0, 3, 2, 1]]  # 顺时针排序

			rescaled_geo = geo / [width*4/int(batch['shape'][0][1]), height*4/int(batch['shape'][0][0])]
			boxes.append(rescaled_geo.astype(np.int16))
	boxes = np.array(boxes).astype(np.int16)
	boxes_batch.append(boxes)


	raw_metric = QuadMeasurer().validate_measure(
		batch, [boxes_batch], False)

	raw_metrics.append(raw_metric)

metrics = QuadMeasurer().gather_measure(raw_metrics)

for key, metric in metrics.items():
	print('%s : %f (%d)' % (key,metric.avg, metric.count))




