import torch
from torch import nn

from src.utils import sisnr_utls


class SiSNRLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        source_1: torch.Tensor,
        source_2: torch.Tensor,
        predicted_source_1: torch.Tensor,
        predicted_source_2: torch.Tensor,
        **kwargs
    ):
        """
        Calculates SI-SNR for predicted and ground truth signals
        (get maximum for all permutations)

        Args:
            source_1 (torch.Tensor): ground truth signal 1
            source_2 (torch.Tensor): ground truth signal 2
            predicted_source_1 (torch.Tensor): predicted signal 1
            predicted_source_2 (torch.Tensor): predicted signal 2
        Returns:
            loss_dict (dict[str, torch.Tensor]): dict containing loss
        """
        loss = sisnr_utls.compute_pair_sisnr(
            predicted_1=predicted_source_1,
            predicted_2=predicted_source_2,
            target_1=source_1,
            target_2=source_2,
        )

        return {"loss": -loss.mean()}
