import numpy as np
import cv2
import os
from utils.preprocess import reorder_vertexes



class MakeLabel:

    def __init__(self, min_text_size=8,shrink_ratio=0.3):
        self.min_text_size = min_text_size
        self.shrink_ratio = shrink_ratio
        self.reorder_vertexes = reorder_vertexes

    def point_inside_of_quad(self, px, py, quad_xy_list, p_min, p_max):
        # 点（px,py）对应直线v0-v1在x=px处y的值
        # [(x0,y0),(x1,y1),(px,py)]三点若为顺时针，则点在内部，可根据行列式的性质求三角形面积
        if (p_min[0] <= px <= p_max[0]) and (p_min[1] <= py <= p_max[1]):
            # xy_list[0] = [x1-x0, y1-y0]
            xy_list = np.zeros((4, 2))
            xy_list[:3, :] = quad_xy_list[1:4, :] - quad_xy_list[:3, :]
            xy_list[3] = quad_xy_list[0, :] - quad_xy_list[3, :]
            # yx_list[0] = [y0, x0]
            yx_list = np.zeros((4, 2))
            yx_list[:, :] = quad_xy_list[:, -1:-3:-1]
            # a[0] = [x1-x0, y1-y0] * ([py, px] - [y0, x0])
            #    = [x1-x0, y1-y0] * [py-y0, px-x0]
            #    = [(x1-x0)*(py-y0), (y1-y0)*(px-x0)]
            a = xy_list * ([py, px] - yx_list)
            b = a[:, 0] - a[:, 1]
            if np.amin(b) >= 0 or np.amax(b) <= 0:
                return True
            else:
                return False
        else:
            return False

    def point_inside_of_nth_quad(self, px, py, xy_list, shrink_1, long_edge):
        nth = -1
        vs = [[[0, 0, 3, 3, 0], [1, 1, 2, 2, 1]],
              [[0, 0, 1, 1, 0], [2, 2, 3, 3, 2]]]
        for ith in range(2):
            quad_xy_list = np.concatenate((
                np.reshape(xy_list[vs[long_edge][ith][0]], (1, 2)),
                np.reshape(shrink_1[vs[long_edge][ith][1]], (1, 2)),
                np.reshape(shrink_1[vs[long_edge][ith][2]], (1, 2)),
                np.reshape(xy_list[vs[long_edge][ith][3]], (1, 2))), axis=0)
            p_min = np.amin(quad_xy_list, axis=0)
            p_max = np.amax(quad_xy_list, axis=0)
            if self.point_inside_of_quad(px, py, quad_xy_list, p_min, p_max):
                if nth == -1:
                    nth = ith
                else:
                    nth = -1
                    break
        return nth


    def validate_polygons(self, polygons, ignore_tags, h, w):
        '''
        polygons (numpy.array, required): of shape (num_instances, num_points, 2)
        '''
        if len(polygons) == 0:
            return polygons, ignore_tags
        assert len(polygons) == len(ignore_tags)
        for polygon in polygons:
            polygon[:, 0] = np.clip(polygon[:, 0], 0, w - 1)
            polygon[:, 1] = np.clip(polygon[:, 1], 0, h - 1)

        for i in range(len(polygons)):
            area = self.polygon_area(polygons[i])
            if abs(area) < 1:
                ignore_tags[i] = True
            if area > 0:# 逆时针
                polygons[i] = polygons[i][::-1, :]
        return polygons, ignore_tags


    def polygon_area(self, polygon):
        edge = 0
        for i in range(polygon.shape[0]):
            next_index = (i + 1) % polygon.shape[0]
            edge += (polygon[next_index, 0] - polygon[i, 0]) * (polygon[next_index, 1] - polygon[i, 1])

        return edge / 2.


    def shrink(self, xy_list, ratio=0.3, h_t_shrink=False):
        if ratio == 0.0:
            return xy_list, xy_list
        # 1 计算相邻两点间的距离
        diff_1to3 = xy_list[:3, :] - xy_list[1:4, :]
        diff_4 = xy_list[3:4, :] - xy_list[0:1, :]
        diff = np.concatenate((diff_1to3, diff_4), axis=0)
        # 计算各顶点之间的距离，v0->v1:d0 v1->v2:d1 v2->v3:d2 v3->v0:d3
        # dis = [d0 d1 d2 d3]
        # d = sqrt(dx^2 + dy^2)
        dis = np.sqrt(np.sum(np.square(diff), axis=-1))
        # determine which are long or short edges
        # 标记长短边  [d0+d2, d1+d3]
        # long_edge=0 标识 w > h;扁平的 long_edge=1 标识 w < h 瘦高的
        dis_tmp = np.sum(np.reshape(dis, (2, 2)), axis=0)
        long_edge = int(np.argmax(dis_tmp))
        short_edge = 1 - long_edge
        # cal r length array
        # 3 选择缩进参考距离
        # r = [r0, r1, r2, r3] = [d3, d1, d1, d3] 取短边的长度
        r = [np.minimum(dis[i], dis[(i + 1) % 4]) for i in range(4)]
        # cal theta array
        diff_abs = np.abs(diff)
        diff_abs[:, 0] += 1e-4
        theta = np.arctan(diff_abs[:, 1] / diff_abs[:, 0])
        # shrink two long edges
        temp_new_xy_list = np.copy(xy_list)

        if h_t_shrink:
            wdh = dis_tmp[long_edge] / dis_tmp[short_edge]
            if wdh > 5.0:
                ratio = ratio * (dis_tmp[long_edge] / dis_tmp[short_edge] / 5.0)

        self.shrink_edge(xy_list, temp_new_xy_list, long_edge, r, theta, ratio)
        self.shrink_edge(xy_list, temp_new_xy_list, long_edge + 2, r, theta, ratio)

        # shrink two short edges
        new_xy_list = np.copy(temp_new_xy_list)
        self.shrink_edge(temp_new_xy_list, new_xy_list, short_edge, r, theta, ratio)
        self.shrink_edge(temp_new_xy_list, new_xy_list, short_edge + 2, r, theta, ratio)

        return temp_new_xy_list, new_xy_list, long_edge


    def shrink_edge(self, xy_list, new_xy_list, edge, r, theta, ratio=0.3):
        if ratio == 0.0:
            return
        start_point = edge
        end_point = (edge + 1) % 4
        # print('start_point',start_point,'end_point',end_point)
        # x方向上，符号判断 (x1 - x0) 正负，大小
        long_start_sign_x = np.sign(
            xy_list[end_point, 0] - xy_list[start_point, 0])
        # x0 + dx;  dx = 0.2 * r0 * cos(theta0);   r0*cos(theta0)表示r0在x轴上的投影
        new_xy_list[start_point, 0] = \
            xy_list[start_point, 0] + \
            long_start_sign_x * ratio * r[start_point] * np.cos(theta[start_point])
        # y方向上，符号判断 (y1 - y0)
        long_start_sign_y = np.sign(
            xy_list[end_point, 1] - xy_list[start_point, 1])
        # y0 + dy;  dy = 0.2 * r0 * sin(theta0);   r0*sin(theta0)表示r0在y轴上的投影
        new_xy_list[start_point, 1] = \
            xy_list[start_point, 1] + \
            long_start_sign_y * ratio * r[start_point] * np.sin(theta[start_point])

        # long edge one, end point
        long_end_sign_x = -1 * long_start_sign_x
        new_xy_list[end_point, 0] = \
            xy_list[end_point, 0] + \
            long_end_sign_x * ratio * r[end_point] * np.cos(theta[start_point])
        long_end_sign_y = -1 * long_start_sign_y
        new_xy_list[end_point, 1] = \
            xy_list[end_point, 1] + \
            long_end_sign_y * ratio * r[end_point] * np.sin(theta[start_point])


    def __call__(self, data):
        polygons = data['polygons']
        ignore_tags = data['ignore_tags']
        image = data['image']
        filename = data['filename']

        h, w = image.shape[:2]
        if data['is_training']:
            polygons, ignore_tags = self.validate_polygons(
                polygons, ignore_tags, h, w)
        gt = np.zeros((8, h//4, w//4), dtype=np.float32)
        mask = np.ones((h//4, w//4), dtype=np.float32)

        for i in range(len(polygons)):
            polygon = polygons[i]
            # print(polygon)
            height = max(polygon[:, 1]) - min(polygon[:, 1])
            width = max(polygon[:, 0]) - min(polygon[:, 0])


            if ignore_tags[i] or min(height, width) < self.min_text_size:
                cv2.fillPoly(mask, polygon.astype(
                    np.int32)[np.newaxis, :, :], 0)
                ignore_tags[i] = True
            else:
                polygon = self.reorder_vertexes(polygon)
                _, shrink_xy_list, _ = self.shrink(polygon, 0.3)
                shrink_1, _, long_edge = self.shrink(polygon, 0.6, h_t_shrink=True)  # 缩小头和尾
                p_min = np.amin(shrink_xy_list, axis=0)  # 最小的x,y
                p_max = np.amax(shrink_xy_list, axis=0)  # 最大的x,y
                # floor of the float
                ji_min = (p_min / 4.0 - 0.5).astype(int) - 1
                # +1 for ceil of the float and +1 for include the end
                ji_max = (p_max / 4.0 - 0.5).astype(int) + 3  # ??
                # 坐标值不能小于零 不能大于宽和高
                imin = np.maximum(0, ji_min[1])
                imax = np.minimum(h // 4, ji_max[1])
                jmin = np.maximum(0, ji_min[0])
                jmax = np.minimum(w // 4, ji_max[0])

                # # 遍历所有像素值
                for i in range(int(imin), int(imax)):
                    for j in range(int(jmin), int(jmax)):
                        px = (j + 0.5) * 4.0
                        py = (i + 0.5) * 4.0
                        if self.point_inside_of_quad(px, py,
                                                shrink_xy_list, p_min, p_max):
                            gt[0, i, j] = 1  # 是否为内部像素

                            ith = self.point_inside_of_nth_quad(px, py,
                                                           polygon,
                                                           shrink_1,
                                                           long_edge)
                            vs = [[[3, 0], [1, 2]], [[0, 1], [2, 3]]]
                            if ith in range(2):
                                # gt[i, j, 1] = 1 # 是否为头尾像素
                                if ith == 0:
                                    gt[1, i, j] = 1  # 是否为头
                                else:
                                    gt[2, i, j] = 1  # 是否为尾
                                # gt[i, j, 2:3] = ith #是头还是尾
                                gt[3:5, i, j] = \
                                    polygon[vs[long_edge][ith][0]] - [px, py]  # 头（尾）元素坐标相对顶点的位移
                                gt[5:7, i, j] = \
                                    polygon[vs[long_edge][ith][1]] - [px, py]

        gt[-1, :, :] = mask
        data.update(image=image,
                    polygons=polygons,
                    gt=gt, filename=filename)


        return data


