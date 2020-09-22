import torch
import torch.nn as nn
import cfg


from .dice_loss import DiceLoss
from .l1_loss import MaskL1Loss
from .balance_cross_entropy_loss import BalanceCrossEntropyLoss
from .focal_loss import FocalLoss


class Loss(nn.Module):
    '''
    Balanced CrossEntropy Loss on `binary`,
    MaskL1Loss on `thresh`,
    DiceLoss on `thresh_binary`.
    Note: The meaning of inputs can be figured out in `SegDetectorLossBuilder`.
    '''

    def __init__(self, eps=1e-6):
        super(Loss, self).__init__()
        self.dice_loss = DiceLoss(eps=eps)
        self.l1_loss = MaskL1Loss()
        self.bce_loss = BalanceCrossEntropyLoss()
        self.focal_loss = FocalLoss()

    def smooth_l1_loss(self, prediction_tensor, target_tensor, weights):
        n_q = torch.reshape(self.quad_norm(target_tensor), weights.size())
        diff = prediction_tensor - target_tensor
        abs_diff = torch.abs(diff)
        abs_diff_lt_1 = torch.lt(abs_diff, 1)  # 小于1
        pixel_wise_smooth_l1norm = (torch.sum(
            torch.where(abs_diff_lt_1, 0.5 * torch.pow(abs_diff, 2), abs_diff - 0.5), 1) / n_q) * weights
        return pixel_wise_smooth_l1norm

    def quad_norm(self, g_true):  # 4*nq,nq为短边长度
        t_shape = g_true.permute(0, 2, 3, 1)  # 维度换位 [b,w,h,4]
        shape = t_shape.size()  # n h w c
        delta_xy_matrix = torch.reshape(t_shape, [-1, 2, 2])  # [n*h*w,2,2]
        diff = delta_xy_matrix[:, 0:1, :] - delta_xy_matrix[:, 1:2, :]  # [n*w*h,1,2]
        square = torch.pow(diff, 2)  # # [n*w*h,1,2]

        distance = torch.sqrt(torch.sum(square, 2))  # 按第二个维度求和 [n*w*h,1]
        distance *= 4.0
        distance += cfg.epsilon
        return torch.reshape(distance, shape[:-1])

    def forward(self, gt, pred):

        logits = pred[:, :1, :, :]
        labels = gt[:, :1, :, :]
        # dice_loss = self.dice_loss(predicts, labels, gt[:, -1, :, :])
        beta = 1 - torch.mean(labels)
        # first apply sigmoid activation
        predicts = torch.sigmoid(logits)
        # log +epsilon for stable cal
        inside_score_loss = torch.mean(
            -1 * (beta * labels * torch.log(predicts + cfg.epsilon) +
                  (1 - beta) * (1 - labels) * torch.log(1 - predicts + cfg.epsilon)))
        inside_score_loss *= cfg.lambda_inside_score_loss

        vertex_logits = pred[:, 1:3, :, :]
        vertex_labels = gt[:, 1:3, :, :]

        # t = gt[:, 1, :, :] + gt[:, 2, :, :]
        # one = torch.ones_like(t)
        # t = torch.where(t <= 1, t, one)

        vertex_beta = 1 - (torch.mean(gt[:, 1:2, :, :])
                           / (torch.mean(labels) + cfg.epsilon))
        vertex_predicts = torch.sigmoid(vertex_logits)
        pos = -1 * vertex_beta * vertex_labels * torch.log(vertex_predicts +
                                                           cfg.epsilon)
        neg = -1 * (1 - vertex_beta) * (1 - vertex_labels) * torch.log(
            1 - vertex_predicts + cfg.epsilon)

        # positive_weights = torch.cast(torch.eq(y_true[:, :, :, 0], 1), tf.float32)
        positive_weights = torch.eq(gt[:, 0, :, :], 1).float()
        side_vertex_code_loss = \
            torch.sum(torch.sum(pos + neg, 1) * positive_weights) / (
                    torch.sum(positive_weights) + cfg.epsilon)

        # side_vertex_code_loss = FocalLoss()(vertex_logits, vertex_labels)
        side_vertex_code_loss *= cfg.lambda_side_vertex_code_loss

        g_hat = pred[:, 3:7, :, :]
        g_true = gt[:, 3:7, :, :]

        vertex_weights = torch.eq(gt[:, 1, :, :], 1).float()
        # vertex_weights = torch.eq(t, 1).float()
        pixel_wise_smooth_l1norm = self.smooth_l1_loss(g_hat, g_true, vertex_weights)
        side_vertex_coord_loss = torch.sum(pixel_wise_smooth_l1norm) / (
				torch.sum(vertex_weights) + cfg.epsilon)
        side_vertex_coord_loss *= cfg.lambda_side_vertex_coord_loss


        print('dice_loss is {:.4f}\t side_vertex_code_loss is {:.4f}\t side_vertex_coord_loss loss is {:.4f}'.format(
				inside_score_loss, side_vertex_code_loss,side_vertex_coord_loss))
        return inside_score_loss + side_vertex_code_loss + side_vertex_coord_loss









