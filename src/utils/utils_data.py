from typing import Any, Dict, NoReturn, Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


class MNISTDataset(Dataset):
    def __init__(self, file_name: str) -> NoReturn:
        self.df = pd.read_csv(file_name)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        pixels = []
        for i in range(784):
            pixel = self.df.iloc[idx][[f"pixel{str(i)}"]].item()
            pixels.append(pixel)
        pixels = torch.FloatTensor(pixels).reshape(1, 28, 28)
        label = torch.LongTensor([self.df.iloc[idx]["label"].item()])
        return pixels, label


def prepare_loader(
    path_data: str, loader_kwargs: Dict[str, Any]
) -> DataLoader:
    dataset = MNISTDataset(file_name=path_data)
    dataloader = DataLoader(dataset, **loader_kwargs)
    return dataloader


def prepare_train_test_loaderes(
    path_train_split: str,
    path_test_split: str,
    train_kwargs: Dict[str, Any],
    test_kwargs: Dict[str, Any],
) -> Tuple[DataLoader, DataLoader]:
    train_loader = prepare_loader(
        path_data=path_train_split, loader_kwargs=train_kwargs
    )
    test_loader = prepare_loader(
        path_data=path_test_split, loader_kwargs=test_kwargs
    )
    return train_loader, test_loader
