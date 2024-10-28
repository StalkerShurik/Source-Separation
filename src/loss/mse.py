import torch
from torch import nn


class SignalMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss()

    def forward(
        self,
        source_1: torch.Tensor,
        source_2: torch.Tensor,
        predicted_source_1: torch.Tensor,
        predicted_source_2: torch.Tensor,
        **kwargs
    ):
        """
        Calculates MSE for predicted and ground truth signals.

        Args:
            source_1 (torch.Tensor): ground truth signal 1
            source_2 (torch.Tensor): ground truth signal 2
            predicted_source_1 (torch.Tensor): predicted signal 1
            predicted_source_2 (torch.Tensor): predicted signal 2
        Returns:
            loss_dict (dict[str, torch.Tensor]): dict containing loss
        """
        l1 = self.loss(source_1, predicted_source_1) + self.loss(
            source_2, predicted_source_2
        )
        l2 = self.loss(source_1, predicted_source_2) + self.loss(
            source_2, predicted_source_1
        )
        return {"loss": l1 if l1 < l2 else l2}
