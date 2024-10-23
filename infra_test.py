from torch.utils.data import DataLoader

from src.datasets import SourceSeparationDataset

# from src.datasets.data_utils import get_dataloaders


dataset_train = SourceSeparationDataset(part="train")
dataset_val = SourceSeparationDataset(part="val")
dataset_test = SourceSeparationDataset(part="test")

dataloader_train = DataLoader(dataset_train, batch_size=2)
dataloader_test = DataLoader(dataset_test)
dataloader_val = DataLoader(dataset_val)

for batch in dataloader_train:
    print(batch)

    print(len(batch["mix"]))
    print(len(batch["source1"]))
    print(len(batch["source2"]))

    break

for batch in dataloader_test:
    print(batch)

    print(len(batch["mix"]))
    print(len(batch["source1"]))
    print(len(batch["source2"]))

    break
