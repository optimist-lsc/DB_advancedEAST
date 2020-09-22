import imgaug.augmenters as iaa
import imgaug

import cv2
import math


class AugmentData:

    def __init__(self, size=(512, 512),keep_ratio=False,only_resize=False):
        self.size = size
        self.keep_ratio = keep_ratio
        self.only_resize = only_resize
        self.augmenter = iaa.Sequential([
            iaa.Fliplr(0.5),
            iaa.Affine(rotate=(-10, 10)),
            iaa.Resize((0.5, 3.0))
        ])

    def may_augment_annotation(self, aug, data, shape):
        if aug is None:
            return data

        line_polys = []
        for line in data['lines']:
            if self.only_resize:
                new_poly = [(p[0], p[1]) for p in line['poly']]
            else:
                new_poly = self.may_augment_poly(aug, shape, line['poly'])  # 变换后得图片和坐标
            line_polys.append({
                'points': new_poly,
                'ignore': line['text'] == '###',
                'text': line['text'],
            })
        data['polys'] = line_polys
        return data

    def may_augment_poly(self, aug, img_shape, poly):
        keypoints = [imgaug.Keypoint(p[0], p[1]) for p in poly]
        keypoints = aug.augment_keypoints(
            [imgaug.KeypointsOnImage(keypoints, shape=img_shape)])[0].keypoints
        poly = [(p.x, p.y) for p in keypoints]
        return poly

    def resize_image(self, image):
        origin_height, origin_width, _ = image.shape
        height = self.size[0]
        width = self.size[1]
        if self.keep_ratio:
            # width = origin_width * height / origin_height
            height = origin_height * width / origin_width
            # N = math.ceil(width / 32)
            N = math.floor(height / 32)
            # width = N * 32
            height = N * 32
        image = cv2.resize(image, (width, height))
        return image

    def __call__(self, data):
        image = data['image']
        aug = None
        shape = image.shape

        if self.augmenter:
            aug = self.augmenter.to_deterministic()
            if self.only_resize:
                data['image'] = self.resize_image(image)
            else:
                data['image'] = aug.augment_image(image)
            self.may_augment_annotation(aug, data, shape)

        filename = data.get('filename', data.get('data_id', ''))
        data.update(filename=filename, shape=shape[:2]) # shape在这里更新
        if not self.only_resize:
            data['is_training'] = True 
        else:
            data['is_training'] = False
        return data





