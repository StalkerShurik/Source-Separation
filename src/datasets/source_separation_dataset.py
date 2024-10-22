from src.datasets.base_dataset import BaseDataset

from src.utils.io_utils import ROOT_PATH

from pathlib import Path
import os
import wget
import json
from tqdm import tqdm

import torchaudio

data_dir = "dla_datasets"

class SourceSeparationDataset(BaseDataset):

    def __init__(self, part, data_dir=None, *args, **kwargs):
        if type(part) != type(""):
            raise TypeError("part must be string")

        if part not in ["train", "val", "test"]:
            raise ValueError("only train, val, test parts are supported")
        
        if data_dir is None:
            data_dir = ROOT_PATH / "dla_dataset" 

        self._data_dir = data_dir

        index = self._get_or_load_index(part)

        super().__init__(index, *args, **kwargs)

    def _get_or_load_index(self, part):
        index_path = self._data_dir / f"{part}_index.json"
        if index_path.exists():
            with index_path.open() as f:
                index = json.load(f)
        else:
            index = self._create_index(part)
            with index_path.open("w") as f:
                json.dump(index, f, indent=2)
        return index

    def _create_index(self, part):
        index = []
        split_dir = self._data_dir / "audio" / part
        if not split_dir.exists():
            raise Exception("Wrong dataset structure. It must contain audio folder with at least one of train/test/val folders")
        
        mix_dir = split_dir / "mix"

        if not split_dir.exists():
            raise Exception(f"Wrong dataset structure. {part} must contain mix folder")

        mix_paths = set()

        for dirpath, dirnames, filenames in os.walk(str(mix_dir)):
                mix_paths = filenames

        mouths_dir = self._data_dir / "mouths"

        if not mouths_dir.exists():
            raise Exception(f"{mouths_dir} doesnt exists")

        if part == "test":
            for path in mix_paths:
                # mouth_path = mouths_dir / (path[:-3] + "npz")
                # if not mouth_path.exists():
                #     raise Exception(f"{mouth_path} doesnt exists")
                index.append({
                    "mix": str(mix_dir / path),
                    #"mouth": str(mouth_path)
                })
        else:
             for path in mix_paths:
                s1_path = split_dir / "s1" / path
                s2_path = split_dir / "s2" / path
                # mouth_path = mouths_dir / (path[:-3] + "npz")
                if not s1_path.exists():
                    raise Exception(f"{s1_path} doesnt exists")

                if not s2_path.exists():
                    raise Exception(f"{s2_path} doesnt exists")

                # if not mouth_path.exists():
                #     raise Exception(f"{mouth_path} doesnt exists")

                index.append({
                    "mix": str(mix_dir / path),
                    # "mouth": str(mouth_path),
                    "s1": str(s1_path),
                    "s2": str(s2_path)
                })
        return index






