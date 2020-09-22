import torch

from torch import nn
from datasets.image_dataset import ImageDataset
from datasets.data_loader import DataLoader
from model.DB_model import EAST
from loss.DB_loss import Loss
import os
import time
import numpy as np
from tqdm import tqdm

import cfg
from utils.nms_v2 import nms
from utils.lr_scheduler import LR_Scheduler_Head
from datasets.make_icdar_data import ICDARCollectFN
from utils.preprocess import reorder_vertexes
from concern.quad_measurer import QuadMeasurer
from utils.log import Logger

if __name__ == '__main__':

	logger = Logger('saved_model')

	torch.backends.cudnn.benchmark = True

	trainset = ImageDataset(cfg.train_image_dir,cfg.train_label_dir,cfg.input_size)
	valset = ImageDataset(cfg.val_image_dir,cfg.val_label_dir)

	train_loader = DataLoader(trainset,cfg.batch_size,num_workers=2,is_train=True,collect_fn=None)
	val_loader = DataLoader(valset,batch_size=1,num_workers=0,is_train=False,collect_fn=ICDARCollectFN())

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = EAST()

	model = nn.DataParallel(model)

	if cfg.load_weights and os.path.exists(cfg.saved_model_weights_file_path):
		print('loading %s' % cfg.saved_model_weights_file_path)
		model.load_state_dict(torch.load(cfg.saved_model_weights_file_path))

	model.to(device)
	criterion = Loss()
	criterion = nn.DataParallel(criterion)
	criterion.to(device)

	optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.decay)

	scheduler = LR_Scheduler_Head('poly', cfg.lr, cfg.epoch_num, len(train_loader),warmup_epochs=5)


	for epoch in range(cfg.initial_epoch+1, cfg.epoch_num+1):
		model.train()
		epoch_time = time.time()
		for i, (batch) in enumerate(train_loader):
			optimizer.zero_grad()
			start_time = time.time()
			img, gt_map = batch['image'].to(device),batch['gt'].to(device)
			east_detect = model(img,True)
			loss = criterion(gt_map, east_detect).mean()
			loss.backward()
			scheduler(optimizer, i, epoch)
			optimizer.step()

			print('Epoch is [{}/{}], mini-batch is [{}/{}], time consumption is {:.8f}, batch_loss is {:.8f}'.format(\
			  epoch, cfg.epoch_num, i+1, int(len(train_loader)), time.time()-start_time, loss.item()))


		if epoch % cfg.val_epoch==0:
			model.eval()
			raw_metrics = []
			for i, (batch) in tqdm(enumerate(val_loader)):
				img = batch['image'].to(device)
				with torch.no_grad():
					east_detect = model(img,False)
				# east_detect[:, :3, :, :] = torch.sigmoid(east_detect[:, :3, :, :])
				y = torch.squeeze(east_detect, axis=0)  # [1,7,w,h]转[7,w,h] 删除单维度条目
				y = y.cpu().numpy()
				height, width = y.shape[1:]
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
				logger.metrics('%d : %s : %f (%d)' % (epoch, key, metric.avg, metric.count))



			state_dict = model.state_dict()

			torch.save(state_dict, 'model/resnest_east_weights_{}_{}.h5'.format(\
				  cfg.train_task_id, epoch))


