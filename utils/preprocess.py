import numpy as np

import cfg



def batch_reorder_vertexes(xy_list_array):
    reorder_xy_list_array = np.zeros_like(xy_list_array)
    for xy_list, i in zip(xy_list_array, range(len(xy_list_array))):
        reorder_xy_list_array[i] = reorder_vertexes(xy_list)
    return reorder_xy_list_array


def reorder_vertexes(xy_list):
    reorder_xy_list = np.zeros_like(xy_list)
    # determine the first point with the smallest x,
    # if two has same x, choose that with smallest y,
    ordered = np.argsort(xy_list, axis=0)   # 在列的维度上将各行的元素升序排列，返回索引
    xmin1_index = ordered[0, 0] # 取ordered第一列的前两个值，即前两个x的最小值（索引号）
    xmin2_index = ordered[1, 0]
    if xy_list[xmin1_index, 0] == xy_list[xmin2_index, 0]:  # 如果xy_list第一列的前两个值相等...
        if xy_list[xmin1_index, 1] <= xy_list[xmin2_index, 1]:  # ...比较第二列的两个值
            reorder_xy_list[0] = xy_list[xmin1_index]
            first_v = xmin1_index
        else:
            reorder_xy_list[0] = xy_list[xmin2_index]
            first_v = xmin2_index
    else:
        reorder_xy_list[0] = xy_list[xmin1_index]
        first_v = xmin1_index
    # connect the first point to others, the third point on the other side of
    # the line with the middle slope
    others = list(range(4)) # 记录剩余点的索引
    others.remove(first_v)  # 拿掉已经确定的索引
    k = np.zeros((len(others),))
    for index, i in zip(others, range(len(others))):
        # (y_i - y_1) / (x_i - x_1) 斜率
        k[i] = (xy_list[index, 1] - xy_list[first_v, 1]) \
                    / (xy_list[index, 0] - xy_list[first_v, 0] + cfg.epsilon)
    k_mid = np.argsort(k)[1]    # 斜率的中位数所对的索引号
    third_v = others[k_mid] # 由这个索引号找到第三个顶点
    reorder_xy_list[2] = xy_list[third_v]
    # determine the second point which on the bigger side of the middle line
    others.remove(third_v)
    b_mid = xy_list[first_v, 1] - k[k_mid] * xy_list[first_v, 0]    # 计算直线的截距
    second_v, fourth_v = 0, 0
    for index, i in zip(others, range(len(others))):
        # delta = y - (k * x + b)
        delta_y = xy_list[index, 1] - (k[k_mid] * xy_list[index, 0] + b_mid)
        if delta_y > 0:
            second_v = index
        else:
            fourth_v = index
    reorder_xy_list[1] = xy_list[second_v]
    reorder_xy_list[3] = xy_list[fourth_v]
    # compare slope of 13 and 24, determine the final order
    # 万一第一步搞错了，由最小的  坐标确定的点不是v1而是v4，那就把v4换到v1的位置，然后顺时针依次移动其他点。
    k13 = k[k_mid]
    k24 = (xy_list[second_v, 1] - xy_list[fourth_v, 1]) / (
                xy_list[second_v, 0] - xy_list[fourth_v, 0] + cfg.epsilon)
    if k13 < k24:
        tmp_x, tmp_y = reorder_xy_list[3, 0], reorder_xy_list[3, 1]
        for i in range(2, -1, -1):
            reorder_xy_list[i + 1] = reorder_xy_list[i]
        reorder_xy_list[0, 0], reorder_xy_list[0, 1] = tmp_x, tmp_y
    return reorder_xy_list


def resize_image(im, max_img_size=cfg.input_size):
    im_width = np.minimum(im.width, max_img_size)#####min(475,256)=256
    if im_width == max_img_size < im.width:
        im_height = int((im_width / im.width) * im.height)####256/475 * 300==161
    else:
        im_height = im.height#### 300
    o_height = np.minimum(im_height, max_img_size) ####min(161,256)===161
    if o_height == max_img_size < im_height:
        o_width = int((o_height / im_height) * im_width)
    else:
        o_width = im_width###256
    d_wight = o_width - (o_width % 32)####256-0=250
    d_height = o_height - (o_height % 32)#####161-1=160
    return d_wight, d_height

