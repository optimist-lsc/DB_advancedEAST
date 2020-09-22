import torch
import torch.nn.functional as F
import torch.nn as nn

class FocalLoss(nn.Module):
    # 此处alpha直接设为1，不做正负样本的均衡，若想做也行，在forward中写成如下格式即可
    # self.alpha = self.alpha*targets + (1-self.alpha)*(1-targets)
    # 这样当targets为1则取alpha，targets为0则取1-alpha
    def __init__(self, alpha=0.75, gamma=2, logits=True, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        self.alpha = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        # 此处的BCE_Loss可以用-log(pt)代替，它们是等价的
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        # F_loss = torch.sum(F_loss,dim=(2,3)) / torch.sum(inside_pix_labels,dim=(2,3)) # 每张图像的两个通道求均值
        if self.reduce:
            return torch.mean(F_loss)
        else:
            return torch.sum(F_loss)