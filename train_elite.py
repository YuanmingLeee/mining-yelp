import pickle
from datetime import datetime

import numpy as np
import torch
import tqdm as tqdm
from torch import nn
from torch import optim
from torch.nn.modules.loss import _Loss
from torch.optim.optimizer import Optimizer
from torch.utils import data as tdata

from configs import DATA_DIR, OUTPUT_DIR
from data_engine.data_loader import elite_preprocessor, load_data
from data_engine.dataset import EliteDataset
from models.EliteNet import EliteNet
from .helper import get_accuracy

random_seed = 42
split_ratio = .2
bs = 1024
lr = 1e-3
epochs = 100
eps = 1e-8
weight_decay = 1e-6
CSV_PATH = DATA_DIR / 'user-profiling.csv'


def train(net: nn.Module,
          data_loader: tdata.DataLoader,
          optimizer: Optimizer,
          criterion: _Loss,
          data_size: int = None,
          log_batch_num: int = None
          ):
    """Train one epoch

    Args:
        net (nn.Module): model structure
        data_loader (torch.utils.data.DataLoader): data loader
        optimizer (torch.optim.optimizer.Optimizer): optimizer
        data_size (int): total number of data, necessary when using a sampler to split training and validation data.
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
        losses += loss.item()
        accs += acc

        if log_batch_num is not None and batch_num % log_batch_num == 0:
            print('step {:d} | batch loss {:g} | acc {:g}'
                  .format(batch_num, running_loss / log_batch_num, running_accs / len(outputs)))
            running_loss = 0.
            running_accs = 0.

    dataset_size = len(data_loader.dataset) if not data_size else data_size
    return losses / dataset_size, accs / dataset_size


def test(net: nn.Module,
         data_loader: tdata.DataLoader,
         criterion,
         data_size: int = None
         ):
    """Test (validate) one epoch of the given data

    Args:
        net (nn.Module): model structure
        data_loader (torch.utils.data.DataLoader): data loader
        criterion (torch.nn.modules.loss._Loss): loss for test
        data_size (int): total number of data, necessary when using a sampler to split training and validation data.

    Returns:
        Tuple of test loss and test accuracy
    """
    net.eval()

    loss = 0.
    acc = 0.

    for samples in data_loader:
        features = samples['features'].cuda()
        labels = samples['label'].cuda()

        # forward
        output = net(features)
        loss += criterion(output, labels).item()
        acc += get_accuracy(output, labels)

    dataset_size = len(data_loader.dataset) if not data_size else data_size
    return loss / dataset_size, acc / dataset_size


def main():
    # prepare data
    print('Loading data...')
    dataset = EliteDataset(CSV_PATH, preprocessor=elite_preprocessor)
    train_loader, val_loader, (train_size, val_size) = load_data(dataset, split_ratio, bs=bs)
    print('Finish loading')

    # model
    net = EliteNet().cuda()

    # optimizer
    optimizer = optim.Adam(net.parameters(), lr=lr, eps=eps, weight_decay=weight_decay)

    # loss
    loss = nn.CrossEntropyLoss()

    # statistics
    train_losses = np.zeros(epochs, dtype=np.float)
    train_accs = np.zeros(epochs, dtype=np.float)
    val_losses = np.zeros(epochs, dtype=np.float)
    val_accs = np.zeros(epochs, dtype=np.float)
    best_val_loss = float('inf')

    # misc
    name_seed = datetime.now().strftime('%m%d-%H%M%S')

    t = tqdm.trange(epochs)
    for epoch in t:
        train_loss, train_acc = train(net, data_loader=train_loader, optimizer=optimizer, criterion=loss,
                                      data_size=train_size)
        val_loss, val_acc = test(net, data_loader=val_loader, criterion=loss, data_size=val_size)

        # process statistics
        train_losses[epoch], train_accs[epoch] = train_loss, train_acc
        val_losses[epoch], val_accs[epoch] = val_loss, val_acc

        t.set_description('[epoch {:d}] train loss {:g} | acc {:g} || val loss {:g} | acc {:g}'
                          .format(epoch, train_loss, train_acc, val_loss, val_acc))

        # save model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(net.state_dict(), OUTPUT_DIR / 'user-elite-{:s}.pth'.format(name_seed))

    # save statistic
    with open(OUTPUT_DIR / 'user-elite-stat-{:s}.pkl'.format(name_seed), 'wb') as f:
        training_info = {'batch_size': bs, 'epoch': epochs, 'lr': lr, 'weight_decay': weight_decay, 'eps': eps}
        stat = {'train_loss': train_losses, 'train_acc': train_accs, 'val_loss': val_losses, 'val_acc': val_accs}
        content = {'info': training_info, 'stat': stat}
        pickle.dump(content, f)


if __name__ == '__main__':
    main()
