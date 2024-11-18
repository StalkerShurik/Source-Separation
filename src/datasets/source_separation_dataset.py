import json
import typing as tp
from pathlib import Path

import numpy as np
import torch
import torchaudio

from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import DATA_PATH


class SourceSeparationDataset(BaseDataset):
    def __init__(
        self,
        part: str,
        dataset_dir: str | Path = "dla_dataset",
        *args,
        **kwargs,
    ) -> None:
        """
        Args:
            part (str | Path): part of dataset(test/train/val)
            dataset_dir (Path | str): path to dataset
        """
        self._data_dir = DATA_PATH / dataset_dir
        index = self._get_or_load_index(part)
        if part not in ("train", "val", "test"):
            raise ValueError(f"Invalid part {part}")
        self._part = part

        super().__init__(index, *args, **kwargs)

    def __getitem__(self, ind: int) -> dict[str, tp.Any]:
        """
        Get element from the index, preprocess it, and combine it
        into a dict.

        Notice that the choice of key names is defined by the template user.
        However, they should be consistent across dataset getitem, collate_fn,
        loss_function forward method, and model forward method.

        Args:
            ind (int): index in the self.index list.
        Returns:
            instance_data (dict): dict, containing instance
                (a single dataset element).
        """
        data_dict = self._index[ind]
        mix_data = torchaudio.load(data_dict["mix"], backend="soundfile")[0]
        instance_data = {"mix": mix_data}

        npz_video_mouth_1 = np.load(data_dict["mouth1"])
        npz_video_mouth_2 = np.load(data_dict["mouth2"])
        npz_mouth_embed_1 = np.load(data_dict["mouth1_embed"])
        npz_mouth_embed_2 = np.load(data_dict["mouth2_embed"])

        instance_data.update(
            {
                #"video1": torch.tensor(npz_video_mouth_1["data"]).unsqueeze(0),
                #"video2": torch.tensor(npz_video_mouth_2["data"]).unsqueeze(0),
                "video_embed1" : torch.tensor(npz_mouth_embed_1["data"]).unsqueeze(0),
                "video_embed2" : torch.tensor(npz_mouth_embed_2["data"]).unsqueeze(0),
            }
        )

        if self._part != "test":
            source_1 = torchaudio.load(data_dict["s1"], backend="soundfile")
            source_2 = torchaudio.load(data_dict["s2"], backend="soundfile")
            instance_data["target"] = torch.stack((source_1[0], source_2[0]), dim=1)
            assert source_1[1] == source_2[1]

        instance_data = self.preprocess_data(instance_data)
        instance_data["mix"] = instance_data["mix"][0].unsqueeze(0)
        return instance_data

    def _get_or_load_index(self, part: str | Path) -> list[dict[str, tp.Any]]:
        """
        Get index from json file or build it

        Args:
            part (str | Path): part of dataset(test/train/val)
        Returns:
            index (list[dict[str, tp.Any]])
        """
        index_path = self._data_dir / f"{part}_index.json"
        if index_path.exists():
            with index_path.open() as f:
                index = json.load(f)
        else:
            index = self._create_index(part)
            with index_path.open("w") as f:
                json.dump(index, f, indent=2)
        return index

    def _create_index(self, part: str | Path) -> list[dict[str, tp.Any]]:
        """
        Build index for dataset

        Args:
            part (str | Path): part of dataset(test/train/val)
        Returns:
            index (list[dict[str, tp.Any]])
        """
        index = []
        split_dir = self._data_dir / "audio" / part
        mix_dir = split_dir / "mix"

        assert mix_dir.exists()

        for path in sorted(mix_dir.iterdir()):
            assert path.is_file()
            row = {
                "mix": str(path),
            }

            f1, f2 = path.name.split("_")  # get id of speakers
            f2 = f2[:-4]  # remove .wav

            f1 += ".npz"
            f2 += ".npz"

            mouth_path_1 = path.parent.parent.parent.parent / "mouths" / f1
            mouth_path_2 = path.parent.parent.parent.parent / "mouths" / f2
            mouth_embed_path_1 = path.parent.parent.parent.parent / "mouths_embeds" / f1
            mouth_embed_path_2 = path.parent.parent.parent.parent / "mouths_embeds" / f2
            assert mouth_path_1.exists()
            assert mouth_path_2.exists()
            assert mouth_embed_path_1.exists()
            assert mouth_embed_path_2.exists()

            row.update(
                {
                    "mouth1": str(mouth_path_1),
                    "mouth2": str(mouth_path_2),
                    "mouth1_embed": str(mouth_embed_path_1),
                    "mouth2_embed": str(mouth_embed_path_2),
                }
            )

            if part != "test":
                s1_path = split_dir / "s1" / path.name
                s2_path = split_dir / "s2" / path.name
                assert s1_path.is_file() and s2_path.is_file()
                row.update(
                    {
                        "s1": str(s1_path),
                        "s2": str(s2_path),
                    }
                )
            index.append(row)
        return index
