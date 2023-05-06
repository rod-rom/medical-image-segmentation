import torch
from utils import _threshold


class IoU(torch.nn.Module):
    def __init__(self, eps=1e-7, threshold=0.5):
        super(IoU, self).__init__()
        self.eps = eps
        self.threshold = threshold

    def forward(self, inputs, targets):
        assert inputs.size() == targets.size()
        inputs = torch.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        targets = _threshold(targets, self.threshold)
        # intersection is equivalent to True Positive count
        # union is the mutually inclusive area of all labels & predictions
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection + self.eps

        iou = (intersection + self.eps) / union

        return iou
