import torch


def collate_fn(dataset_items: list[dict]) -> dict[str, torch.Tensor]:
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """

    result_batch = {}
    for key in dataset_items[0].keys():
        if isinstance(dataset_items[0][key], torch.Tensor):
            result_batch[key] = torch.stack(
                [sample[key][0] for sample in dataset_items], dim=0
            )
        if isinstance(dataset_items[0][key], str):
            result_batch[key] = [sample[key] for sample in dataset_items]

    result_batch["video_embed"] = torch.concat(
        [result_batch["video_embed1"], result_batch["video_embed2"]], dim=0
    ).squeeze(1)
    result_batch.pop("video_embed1")
    result_batch.pop("video_embed2")
    return result_batch
