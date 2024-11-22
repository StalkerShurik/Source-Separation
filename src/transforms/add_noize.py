import torch
from torch import nn


class AddNoize1D(nn.Module):

    """
    Batch-version of Normalize for 1D Input.
    Used as an example of a batch transform.
    """

    def __init__(self, std=0.1):
        """
        Args:
            mean (float): mean used in the normalization.
            std (float): std used in the normalization.
        """
        super().__init__()

        self.std = std

    def forward(self, x):
        """
        Args:
            x (Tensor): input tensor.
        Returns:
            x (Tensor): normalized tensor.
        """
        noize = torch.rand_like(x) * self.std
        return x + noize
