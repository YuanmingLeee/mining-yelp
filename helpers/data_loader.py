from pathlib import Path
from typing import Callable, Union, IO

import pandas as pd
import torch
from sklearn import preprocessing
from torch.utils import data as tdata


class EliteDataset(tdata.Dataset):
    def __init__(self, path: Union[str, Path, IO], transform: Callable = None):
        dataset = pd.read_csv(path)

        _x = dataset.iloc[:, :-1]
        if transform:
            _x = transform(_x)
        self._x = torch.tensor(_x.values, dtype=torch.float64)
        self._y = torch.tensor(dataset.iloc[:, -1].values, dtype=torch.int32)

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


if __name__ == '__main__':
    from configs import DATA_DIR

    dataset = EliteDataset(DATA_DIR / 'user-profiling.csv', preprocessor)
