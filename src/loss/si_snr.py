import torch
from torch import nn
from torchmetrics.functional.audio.sdr import scale_invariant_signal_distortion_ratio

from src.utils import metric_utls


class SiSNRLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, target: torch.Tensor, predict: torch.Tensor, **kwargs):
        """
        Calculates SI-SNR for predicted and ground truth signals
        (get maximum for all permutations)

        Args:
            target (torch.Tensor): ground truth signal of size B x S x T
            predict (torch.Tensor): predicted signal of size B x S x T
        Returns:
            loss_dict (dict[str, torch.Tensor]): dict containing loss
        """
        loss = metric_utls.compute_metric(
            target=target,
            predict=predict,
            metric=scale_invariant_signal_distortion_ratio,
        )

        return {"loss": -loss.mean()}
