import torch
from pytorch_metric_learning.losses import ArcFaceLoss


class CustomArcFaceLoss(ArcFaceLoss):
    def __init__(self, centroids: torch.Tensor = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.centroids = centroids

    def weight_init_func(self, W: torch.nn.Parameter) -> torch.nn.Parameter:
        """"""
        if self.centroids is not None:
            W.values = self.centroids
        return W
