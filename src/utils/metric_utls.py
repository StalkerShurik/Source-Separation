import torch
import torchmetrics
from torchmetrics import PermutationInvariantTraining

# from torchmetrics.functional.audio.sdr import scale_invariant_signal_distortion_ratio
# from torchmetrics.functional.audio.sdr import signal_distortion_ratio
# from torchmetrics.functional.audio.pesq import perceptual_evaluation_speech_quality
# from torchmetrcis.funtional.audio.stoi import short_time_objective_intelligibility


def compute_metric(
    target: torch.Tensor, predict: torch.Tensor, metric: torchmetrics.functional
) -> torch.Tensor:
    pit = PermutationInvariantTraining(metric_func=metric, eval_func="max")
    return pit(predict, target)
