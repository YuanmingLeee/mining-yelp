import numpy as np
import pandas as pd
import torch.utils.data as tdata
from sklearn import preprocessing


def load_data(dataset: tdata.Dataset, ratio: float, bs: int):
    """Prepare data from torch dataset for training and validation.
    Args:
        dataset (torch.utils.data.Dataset): loaded dataset
        ratio (float): split ratio
        bs (int): batch size

    Returns:
        Tuple of training data loader, validation data loader and
            a tuple of size containing training dataset size and validation
            dataset size respectively
    """
    dataset_size = len(dataset)

    # prepare for shuffle
    indices = np.arange(dataset_size)
    np.random.shuffle(indices)
    split_idx = int(np.floor(ratio * dataset_size))
    train_indices, val_indices = indices[split_idx:], indices[:split_idx]

    # split dataset
    train_sampler = tdata.SubsetRandomSampler(train_indices)
    val_sampler = tdata.SubsetRandomSampler(val_indices)
    train_loader = tdata.DataLoader(dataset, batch_size=bs, sampler=train_sampler)
    val_loader = tdata.DataLoader(dataset, batch_size=bs, sampler=val_sampler)

    return train_loader, val_loader, (len(train_indices), len(val_indices))


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
