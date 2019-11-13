from pathlib import Path
from typing import Callable, Union, IO

import pandas as pd
import torch
from torch.utils import data as tdata
from data_loader import *

class EliteDataset(tdata.Dataset):
    def __init__(self,
                 path: Union[str, Path, IO] = None,
                 df: pd.DataFrame = None,
                 preprocessor: Callable = None):
        # loading dataset
        if df:
            dataset = df
        else:
            dataset = pd.read_csv(path)

        if preprocessor:
            dataset = preprocessor(dataset)
        self._x = torch.tensor(dataset.values[:, :-1], dtype=torch.float)
        self._y = torch.tensor(dataset.iloc[:, -1].values, dtype=torch.long)

    def __len__(self):
        return self._x.shape[0]

    def __getitem__(self, item):
        features = self._x[item]
        label = self._y[item]

        sample = {'features': features, 'label': label}

        return sample

class UsefulDataset(tdata.Dataset):
    def __init__(self, df: pd.DataFrame = None, word2int_mapping):
        dataset = text_preprocess(df, word2int_mapping)
        self._x = torch.tensor(dataset.values[:, :-1], dtype=torch.float)
        self._y = torch.tensor(dataset.iloc[:, -1].values, dtype=torch.long)

    def __len__(self):
        return self._x.shape[0]

    def __getitem__(self, item):
        text = self._x[item]
        label = self._y[item]

        sample = { 'text' : text, 'label': label}
        return sample

class PreNetDataset(tdata.Dataset):
    def __init__(self, path: Union[str, Path, IO], preprocessor, word2int_mapping_path):
        dataset = pd.read_csv(path)

        if preprocessor:
            dataset = preprocessor(dataset)

        # split and pass down
        self.elite_dataset = EliteDataset(dataset.iloc[:, ...])
        self.useful_dataset = 
        self._y = torch.tensor(dataset.iloc[:, -1].values, dtype=torch.long)

    def __len__(self):
        return len(self.elite_dataset)

    def __getitem__(self, item):
        elite_features = self.elite_dataset[item]['features']
        ...
        label = self._y[item]

        samples = {'elite': elite_features, 'text': ..., 'label': label}
