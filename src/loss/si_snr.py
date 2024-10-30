import torch
from torch import nn
from torchmetrics.functional.audio.snr import scale_invariant_signal_noise_ratio

from src.utils import metric_utls


class SiSNRLoss(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.metric = metric_utls.create_permutation_metric(
            metric=scale_invariant_signal_noise_ratio
        )

    def forward(
        self, predict: torch.Tensor, target: torch.Tensor, **kwargs
    ) -> dict[str, torch.Tensor]:
        """
        Calculates SI-SNR for predicted and ground truth signals
        (get maximum for all permutations)

        Args:
            target (torch.Tensor): ground truth signal of size B x S x T
            predict (torch.Tensor): predicted signal of size B x S x T
        Returns:
            loss_dict (dict[str, torch.Tensor]): dict containing loss
        """
        loss = self.metric(predict, target)

        return {"loss": -loss.mean()}
