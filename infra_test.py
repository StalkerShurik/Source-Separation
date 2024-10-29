from torch.utils.data import DataLoader

from src.datasets import SourceSeparationDataset
from src.datasets.collate import collate_fn

# from src.datasets.data_utils import get_dataloaders


dataset_train = SourceSeparationDataset(part="train")
dataset_val = SourceSeparationDataset(part="val")
dataset_test = SourceSeparationDataset(part="test")

dataloader_train = DataLoader(dataset_train, collate_fn=collate_fn, batch_size=2)
dataloader_test = DataLoader(dataset_test, collate_fn=collate_fn)
dataloader_val = DataLoader(dataset_val, collate_fn=collate_fn)

for batch in dataloader_train:
    print(batch)

    print(len(batch["mix"]))
    print(len(batch["source_1"]))
    print(len(batch["source_2"]))

    break

for batch in dataloader_test:
    print(batch)

    print(len(batch["mix"]))

    break
