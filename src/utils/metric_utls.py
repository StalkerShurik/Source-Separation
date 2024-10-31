import typing as tp

import torch


class CustomePIT(torch.nn.Module):  # assume we have only 2 sources
    def __init__(self, metric_func: tp.Callable, **kwargs: tp.Any):
        super().__init__()
        self.metric_func = metric_func
        self.kwargs = kwargs

    def forward(self, predict: torch.Tensor, target: torch.Tensor, **kwargs: tp.Any):
        assert predict.shape == target.shape
        assert predict.ndim == 3  # Batch x Source x Sequence

        metric_perm1 = 0.5 * (
            self.metric_func(predict[:, 0, :], target[:, 0, :], **kwargs).mean()
            + self.metric_func(predict[:, 1, :], target[:, 1, :], **kwargs).mean()
        )
        metric_perm2 = 0.5 * (
            self.metric_func(predict[:, 0, :], target[:, 1, :], **kwargs).mean()
            + self.metric_func(predict[:, 1, :], target[:, 0, :], **kwargs).mean()
        )

        return torch.max(metric_perm1, metric_perm2)
