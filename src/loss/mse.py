import torch
from torch import nn


class SignalMSELoss(nn.Module):
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
        first_order_loss = (
            (source_1 - predicted_source_1) ** 2 + (source_2 - predicted_source_2) ** 2
        ).mean(dim=1)
        second_order_loss = torch.max(
            (source_1 - predicted_source_2) ** 2 + (source_2 - predicted_source_1) ** 2,
        ).mean(dim=1)
        return {"loss": torch.max(first_order_loss, second_order_loss).mean()}
