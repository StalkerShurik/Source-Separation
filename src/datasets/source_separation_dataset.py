import json
import typing as tp
from pathlib import Path

import torchaudio

from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import DATA_PATH


class SourceSeparationDataset(BaseDataset):
    def __init__(
        self, part: str, dataset_dir: str | Path = "dla_dataset", *args, **kwargs
    ) -> None:
        self._data_dir = DATA_PATH / dataset_dir

        index = self._get_or_load_index(part)
        if part not in ("train", "val", "test"):
            raise ValueError(f"Invalid part {part}")
        self._part = part

        super().__init__(index, *args, **kwargs)

    def __getitem__(self, ind: int) -> tp.Dict[str, tp.Any]:
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
        mix_data = torchaudio.load(data_dict["mix"], backend="soundfile")
        instance_data = {"mix": mix_data}

        if self._part != "test":
            instance_data["source1"] = torchaudio.load(
                data_dict["s1"], backend="soundfile"
            )
            instance_data["source2"] = torchaudio.load(
                data_dict["s2"], backend="soundfile"
            )

        instance_data = self.preprocess_data(instance_data)

        return instance_data

    def _get_or_load_index(self, part: str | Path) -> tp.List[tp.Dict[str, tp.Any]]:
        index_path = self._data_dir / f"{part}_index.json"
        if index_path.exists():
            with index_path.open() as f:
                index = json.load(f)
        else:
            index = self._create_index(part)
            with index_path.open("w") as f:
                json.dump(index, f, indent=2)
        return index

    def _create_index(self, part: str | Path) -> tp.List[tp.Dict[str, tp.Any]]:
        index = []
        split_dir = self._data_dir / "audio" / part
        mix_dir = split_dir / "mix"

        assert mix_dir.exists()

        for path in sorted(mix_dir.iterdir()):
            assert path.is_file()
            row = {
                "mix": str(path),
            }
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
