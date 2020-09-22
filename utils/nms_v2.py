# coding=utf-8
import numpy as np
import cv2 as cv

import cfg


def should_merge(region, i, j):
    neighbor = {(i, j - 1)}
    return not region.isdisjoint(neighbor) ##判断region集合是否包含集合neightbor中的元素,不包含返回true ,包含返回false


def region_neighbor(region_set):
    region_pixels = np.array(list(region_set))###由{(a,b)}转换为[[a b]]
    j_min = np.amin(region_pixels, axis=0)[1] - 1  ####取行最小值 ,如不指定，则是所有元素的最大值
    j_max = np.amax(region_pixels, axis=0)[1] + 1
    i_m = np.amin(region_pixels, axis=0)[0] + 1
    region_pixels[:, 0] += 1
    neighbor = {(region_pixels[n, 0], region_pixels[n, 1]) for n in
                range(len(region_pixels))}
    neighbor.add((i_m, j_min))
    neighbor.add((i_m, j_max))
    return neighbor


def region_group(region_list):###len(region_list) =36
    # 将regionlist上下合并为regionGroup
    S = [i for i in range(len(region_list))]
    D = []
    while len(S) > 0:
        m = S.pop(0)
        if len(S) == 0:
            # S has only one element, put it to D
            D.append([m])
        else:
            D.append(rec_region_merge(region_list, m, S))
    return D    # [regionGroup, ....]


def rec_region_merge(region_list, m, S):
    rows = [m]
    tmp = []
    for n in S:
        # 如果regionlist[m]中的所有元素向下平移一个单位，与其他regionlist有重合点，则进行合并
        if not region_neighbor(region_list[m]).isdisjoint(region_list[n]) or \
                not region_neighbor(region_list[n]).isdisjoint(region_list[m]):
            # 第m与n相交
            tmp.append(n)####方法用于在列表末尾添加新的对象
    for d in tmp:
        S.remove(d) ###指定删除list
    for e in tmp:
        rows.extend(rec_region_merge(region_list, e, S))####于在列表末尾一次性追加另一个序列中的多个值


    return rows


def nms(predict, activation_pixels, threshold=cfg.side_vertex_pixel_threshold):
    region_list = []
    # 左右邻接像素集合生成regionList
    for i, j in zip(activation_pixels[0], activation_pixels[1]):
        merge = False
        for k in range(len(region_list)):
            if should_merge(region_list[k], i, j): # 左侧的点是否在regionlist中
                # print(region_list)
                region_list[k].add((i, j))
                merge = True
                # Fixme 重叠文本区域处理，存在和多个区域邻接的pixels，先都merge试试
                # break
        if not merge:
            region_list.append({(i, j)})

    D = region_group(region_list)
    # print(D)
    quad_list = np.zeros((len(D), 4, 2))    # 坐标点
    score_list = np.zeros((len(D), 4))
    for group, g_th in zip(D, range(len(D))):
        total_score = np.zeros((4, 2))
        total_coor = []
        edge_pixes = []
        if len(group)<2:
            continue
        for i in range(len(group)):
            row = group[i]
            for ij in region_list[row]:
                total_coor.append(ij)
                # 判断是否为边界像素
                if tuple((ij[0],ij[1]+1)) not in region_list[row] or tuple((ij[0],ij[1]-1)) not in region_list[row]:
                    edge_pixes.append(tuple((ij[1],ij[0])))

                score = predict[1:3,ij[0], ij[1]]     # 是否是头尾
                # print(score)
                ith = np.argmax(score)
                if score[ith] >= threshold:  #####threshold == 0.9
                    total_score[ith * 2:(ith + 1) * 2] += score[ith]
                    px = (ij[1] + 0.5) * cfg.pixel_size
                    py = (ij[0] + 0.5) * cfg.pixel_size
                    p_v = [px, py] + np.reshape(predict[3:7,ij[0], ij[1]], # (px py)为头部（尾）元素，根据头部（尾）元素预测的偏移量，进行加权平均
                                          (2, 2))
                    quad_list[g_th, ith * 2:(ith + 1) * 2] += score[ith] * p_v

        score_list[g_th] = total_score[:, 0]
        quad_list[g_th] /= (total_score + cfg.epsilon)
        #
        det_h_t = np.greater(score_list[g_th], 0.0) # 头尾是否全部检测出来
        det_h_t_index = np.where(det_h_t)

        if len(det_h_t_index[0])==2 and len(total_coor)>10: # 若只检测出头部或尾部，根据内部像素点的中心值 与 头（尾）的顶点 可以求得另一端的顶点
            h_t_1 = quad_list[g_th][det_h_t_index[0][0]]  # (x,y)
            h_t_2 = quad_list[g_th][det_h_t_index[0][1]]
            short_edge = np.sqrt(np.sum(np.square(h_t_1 - h_t_2)))

            rect = cv.minAreaRect((np.array(edge_pixes).astype(np.float32)+0.5)*cfg.pixel_size)
            rect = np.array(rect)
            rect[1] = tuple(np.array(rect[1])+(short_edge-min(np.array(rect[1]))))
            rect = tuple(rect)
            box = cv.boxPoints(rect)

            quad_list[g_th] = box
            score_list[g_th][(det_h_t_index[0][0]+2)%4] = 1
            score_list[g_th][(det_h_t_index[0][1]+2)%4] = 1

    # print("score_list: ", score_list) # 头部两个顶点+尾部两个顶点
    # print("quad_list: ",quad_list)
    return score_list, quad_list
