import pickle
from datetime import datetime

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.nn.modules.loss import _Loss
from torch.optim.optimizer import Optimizer
from torch.utils import data as tdata

from configs import DATA_DIR, OUTPUT_DIR
from helpers.data_loader import EliteDataset, preprocessor
from helpers.utils import get_accuracy
from models.EliteNet import EliteNet

random_seed = 42
split_ratio = .2
bs = 64
lr = 1e-3
epochs = 50
eps = 1e-8
weight_decay = 1e-6
CSV_PATH = DATA_DIR / 'user-profiling.csv'


def load_data():
    """Prepare data from CSV_PATH for training and validation.

    Returns:
        Tuple of data loader for training and validation
    """
    # get dataset
    dataset = EliteDataset(CSV_PATH, preprocessor)

    dataset_size = len(dataset)

    # prepare for shuffle
    indices = np.arange(dataset_size)
    np.random.shuffle(indices)
    split_idx = int(np.floor(split_ratio * dataset_size))
    train_indices, val_indices = indices[split_idx:], indices[:split_idx]

    # split dataset
    train_sampler = tdata.SubsetRandomSampler(train_indices)
    val_sampler = tdata.SubsetRandomSampler(val_indices)
    train_loader = tdata.DataLoader(dataset, batch_size=bs, sampler=train_sampler)
    val_loader = tdata.DataLoader(dataset, batch_size=bs, sampler=val_sampler)

    return train_loader, val_loader


def train(net: nn.Module, data_loader: tdata.DataLoader, optimizer: Optimizer, criterion: _Loss,
          log_batch_num: int = 1):
    """Train one epoch

    Args:
        net (nn.Module): model structure
        data_loader (torch.utils.data.DataLoader): data loader
        optimizer (torch.optim.optimizer.Optimizer): optimizer
        criterion (torch.nn.modules.loss._Loss): loss for training
        log_batch_num (int): print count, the statistics will print after given number of steps

    Returns:
        Tuple of training loss and training accuracy
    """
    net.train()

    losses = 0.
    accs = 0.
    running_loss = 0.
    running_accs = 0.

    for batch_num, samples in enumerate(data_loader, 0):
        features: torch.Tensor = samples['features'].cuda()
        labels: torch.Tensor = samples['label'].cuda()

        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        acc = get_accuracy(outputs, labels)

        # statistic
        running_loss += loss.item()
        running_accs += acc
        losses += running_loss
        accs += running_accs

        if batch_num % log_batch_num == 0:
            print('step {:d} | batch loss {:.g} | acc {:.g}'
                  .format(batch_num, running_loss / log_batch_num, running_accs / len(outputs)))
            running_loss = 0.
            running_accs = 0.

    return losses, accs / len(data_loader.dataset)


def test(net: nn.Module, data_loader: tdata.DataLoader, criterion):
    """Test (validate) one epoch of the given data

    Args:
        net (nn.Module): model structure
        data_loader (torch.utils.data.DataLoader): data loader
        criterion (torch.nn.modules.loss._Loss): loss for test

    Returns:
        Tuple of test loss and test accuracy
    """
    net.eval()

    loss = 0.
    acc = 0.

    for samples in data_loader:
        features = samples['features'].cuda()
        labels = samples['labels'].cuda()

        # forward
        output = net(features)
        loss += criterion(output, labels).item()
        acc += get_accuracy(output, labels)

    return loss, acc / len(data_loader.dataset)


def main():
    # model
    net = EliteNet().cuda()
    # optimizer
    optimizer = optim.Adam(net.parameters(), lr=lr, eps=eps, weight_decay=weight_decay)
    # loss
    loss = nn.CrossEntropyLoss()
    # prepare data
    train_loader, val_loader = load_data()
    # statistics
    train_losses = np.zeros(epochs, dtype=np.float)
    train_accs = np.zeros(epochs, dtype=np.float)
    val_losses = np.zeros(epochs, dtype=np.float)
    val_accs = np.zeros(epochs, dtype=np.float)
    best_val_loss = float('inf')
    # misc
    name_seed = datetime.now().strftime('%m%d-%H%M%S')

    for epoch in range(epochs):
        train_loss, train_acc = train(net, train_loader, optimizer, loss, log_batch_num=10)
        val_loss, val_acc = test(net, val_loader, loss)

        # process statistics
        train_losses[epoch], train_accs[epoch] = train_loss, train_acc
        val_losses[epoch], val_accs[epoch] = val_loss, val_acc

        print('[epoch {:d}] train loss {:.g} | acc {:.g} || val loss {:.g} | acc {:.g}'
              .format(epoch, train_loss, train_acc, val_loss, val_acc))

        # save model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(net.state_dict(), OUTPUT_DIR / 'user-elite-{:s}'.format(name_seed))

    # save statistic
    with open(OUTPUT_DIR / 'user-elite-stat-{:s}'.format(name_seed)) as f:
        training_info = {'batch_size': bs, 'epoch': epochs, 'lr': lr, 'weight_decay': weight_decay, 'eps': eps}
        stat = {'train_loss': train_losses, 'train_acc': train_accs, 'val_loss': val_losses, 'val_acc': val_accs}
        content = {'info': training_info, 'stat': stat}
        pickle.dump(content, f)
