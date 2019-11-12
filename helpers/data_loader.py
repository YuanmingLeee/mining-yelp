from pathlib import Path
from typing import Callable, Union, IO

import pandas as pd
import torch
from sklearn import preprocessing
from torch.utils import data as tdata


class EliteDataset(tdata.Dataset):
    def __init__(self, path: Union[str, Path, IO], transform: Callable = None):
        dataset = pd.read_csv(path)
        if transform:
            dataset = transform(dataset)
        self._x = torch.tensor(dataset.values[:, :-1], dtype=torch.float)
        self._y = torch.tensor(dataset.iloc[:, -1].values, dtype=torch.long)

    def __len__(self):
        return self._x.shape[0]

    def __getitem__(self, item):
        features = self._x[item]
        label = self._y[item]

        sample = {'features': features, 'label': label}

        return sample


def preprocessor(df: pd.DataFrame):
    # remove unrelated info
    df.drop(columns='user_id', inplace=True)
    # split by label and balance
    elite_df = df.loc[df.elite == 1]
    nonelite_df = df.loc[df.elite == 0].sample(elite_df.shape[0])

    # concatenate and shuffle
    result = pd.concat([elite_df, nonelite_df]).sample(frac=1)

    # clean
    del elite_df, nonelite_df

    # min max scaler
    scaler = preprocessing.MinMaxScaler()
    return pd.DataFrame(scaler.fit_transform(result))
