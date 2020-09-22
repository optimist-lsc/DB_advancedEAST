import argparse
import os
import torch
from torch import nn
from model.DB_model import EAST
import cv2
import math

import numpy as np
from PIL import Image, ImageDraw

import cfg

from utils.preprocess import resize_image,reorder_vertexes
from utils.nms_v2 import nms

def sigmoid(x):
    """`y = 1 / (1 + exp(-x))`"""
    return 1 / (1 + np.exp(-x))

def load_pil(image):
    '''convert PIL Image to torch.Tensor
    '''
    image -= np.array([122.67891434, 116.66876762, 104.00698793])
    image /= 255.
    image = torch.from_numpy(image).permute(2, 0, 1).float()

    return image.unsqueeze(0)

def detect(img_path, model, device,pixel_threshold,quiet=True):
    imgName = os.path.basename(img_path)
    # img = Image.open(img_path)
    # d_wight, d_height = resize_image(img, cfg.max_predict_img_size)
    # img = img.resize((d_wight, d_height), Image.NEAREST).convert('RGB')
    image = cv2.imread(img_path, cv2.IMREAD_COLOR).astype('float32')
    origin_height,origin_width = image.shape[:2]
    height = origin_height * cfg.max_predict_img_size / origin_width
    N = math.floor(height / 32)
    height = N * 32
    img = cv2.resize(image, (cfg.max_predict_img_size, height))
    d_wight, d_height = cfg.max_predict_img_size,height

    with torch.no_grad():
        east_detect=model(load_pil(img).to(device))
    y = np.squeeze(east_detect.cpu().numpy(), axis=0) # [1,7,h,w]转[7,h,w] 删除单维度条目

    # y[:3, :, :] = sigmoid(y[:3, :, :]) # 前三个通道需要sigmod，转为概率
    cond = np.greater_equal(y[0, :, :], pixel_threshold) # 筛选出大于阈值的值,大于阈值为true，小于阈值为false
    activation_pixels = np.where(cond)  # 大于零的坐标点
    quad_scores, quad_after_nms = nms(y, activation_pixels)
    with Image.open(img_path) as im:
        # d_wight, d_height = resize_image(im, cfg.max_predict_img_size)
        scale_ratio_w = d_wight / im.width
        scale_ratio_h = d_height / im.height
        im = im.resize((d_wight, d_height), Image.NEAREST).convert('RGB')
        quad_im = im.copy()
        draw = ImageDraw.Draw(im)
        for i, j in zip(activation_pixels[0], activation_pixels[1]):
            px = (j + 0.5) * cfg.pixel_size
            py = (i + 0.5) * cfg.pixel_size
            line_width, line_color = 1, 'red'
            if y[1,i, j] >= cfg.side_vertex_pixel_threshold:
                # if y[2,i, j] < cfg.trunc_threshold:
                line_width, line_color = 2, 'yellow'
                # elif y[2,i, j] >= 1 - cfg.trunc_threshold:
            elif y[2,i, j] >= cfg.side_vertex_pixel_threshold:
                line_width, line_color = 2, 'green'
            draw.line([(px - 0.5 * cfg.pixel_size, py - 0.5 * cfg.pixel_size),
                       (px + 0.5 * cfg.pixel_size, py - 0.5 * cfg.pixel_size),
                       (px + 0.5 * cfg.pixel_size, py + 0.5 * cfg.pixel_size),
                       (px - 0.5 * cfg.pixel_size, py + 0.5 * cfg.pixel_size),
                       (px - 0.5 * cfg.pixel_size, py - 0.5 * cfg.pixel_size)],
                      width=line_width, fill=line_color)
        act_path = os.path.join('testdata/img_act',imgName)
        im.save(act_path + '_act.jpg')
        quad_draw = ImageDraw.Draw(quad_im)
        txt_items = []
        for score, geo, s in zip(quad_scores, quad_after_nms,
                                 range(len(quad_scores))):

            if np.amin(score) > 0:
                geo = reorder_vertexes(geo)[[0, 3, 2, 1]] # 顺时针排序
                quad_draw.line([tuple(geo[0]),
                                tuple(geo[1]),
                                tuple(geo[2]),
                                tuple(geo[3]),
                                tuple(geo[0])], width=3, fill='red')

                rescaled_geo = geo / [scale_ratio_w, scale_ratio_h]
                rescaled_geo_list = np.reshape(rescaled_geo.astype(np.int32), (8,)).tolist()
                txt_item = ','.join(map(str, rescaled_geo_list))
                txt_items.append(txt_item + '\n')
            elif not quiet:
                print('quad invalid with vertex num less then 4.')
        pre_path = os.path.join('testdata/img_pre',imgName)
        quad_im.save(pre_path + '_predict.jpg')
        txt_path = os.path.join('testdata/txt',imgName)
        if cfg.predict_write2txt and len(txt_items) > 0:
            with open(txt_path[:-4] + '.txt', 'w') as f_txt:
                f_txt.writelines(txt_items)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '-p',
                        default='./testdata/MSRA_img/',
                        help='image path')
    parser.add_argument('--threshold', '-t',
                        default=cfg.pixel_threshold,
                        help='pixel activation threshold')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    path = args.path
    threshold = float(args.threshold)

    model_path='./model/resnest_east_weights_3T800_1100.h5'

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = EAST()
    model = nn.DataParallel(model)
    model.to(device).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    if os.path.isdir(path):
        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)
            detect(img_path, model, device,threshold)
    elif os.path.isfile(path):
        detect(path, model, device, threshold)
    else:
        print('error!')
