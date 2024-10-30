import torch
import torchmetrics
from torchmetrics import PermutationInvariantTraining


def create_permutation_metric(metric: torchmetrics.functional) -> torch.nn.Module:
    return PermutationInvariantTraining(metric_func=metric, eval_func="max")
