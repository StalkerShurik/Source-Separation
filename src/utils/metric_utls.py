import torch
import torchmetrics
from torchmetrics import PermutationInvariantTraining


def compute_metric(
    target: torch.Tensor, predict: torch.Tensor, metric: torchmetrics.functional
) -> torch.Tensor:
    pit = PermutationInvariantTraining(metric_func=metric, eval_func="max")
    return pit(predict, target)
