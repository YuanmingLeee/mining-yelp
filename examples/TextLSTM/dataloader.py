import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def create_dataloader(x_dir, y_dir, batch_size):
    # load data
    x = np.load(x_dir)
    y = np.load(y_dir).squeeze(1)
    
    # create Tensor datasets
    dataset = TensorDataset(torch.from_numpy(x).to(torch.int64).to('cuda'), torch.from_numpy(y).to(torch.long).to('cuda'))
    
    # make sure to SHUFFLE your data
    dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size)
    return dataloader
