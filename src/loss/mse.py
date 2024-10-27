import torch
from torch import nn


class SignalMSELoss(nn.Module):
    """
    Example of a loss function to use.
    """

    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss()

    def forward(
            self, 
            source1: torch.Tensor, 
            source2: torch.Tensor, 
            predicted_source1: torch.Tensor, 
            predicted_source2: torch.Tensor, 
            **batch
        ):
        """
        Loss function calculation logic.

        Note that loss function must return dict. It must contain a value for
        the 'loss' key. If several losses are used, accumulate them into one 'loss'.
        Intermediate losses can be returned with other loss names.

        For example, if you have loss = a_loss + 2 * b_loss. You can return dict
        with 3 keys: 'loss', 'a_loss', 'b_loss'. You can log them individually inside
        the writer. See config.writer.loss_names.

        Args:
            logits (Tensor): model output predictions.
            labels (Tensor): ground-truth labels.
        Returns:
            losses (dict): dict containing calculated loss functions.
        """
        l1 = self.loss(source1, predicted_source1) + self.loss(source2, predicted_source2)
        l2 = self.loss(source1, predicted_source2) + self.loss(source2, predicted_source1)
        return {"loss": l1 if l1 < l2 else l2}
