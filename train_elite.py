import pickle
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm as tqdm
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
bs = 1024
lr = 1e-3
epochs = 100
eps = 1e-8
weight_decay = 1e-6
CSV_PATH = DATA_DIR / 'user-profiling.csv'


def load_data(path: str, ratio: float):
    """Prepare data from CSV_PATH for training and validation.
    Args:
        path (str): dataset file path
        ratio (float): split ratio

    Returns:
        Tuple of training data loader, validation data loader and
            a tuple of size containing training dataset size and validation
            dataset size respectively
    """
    # get dataset
    dataset = EliteDataset(path, preprocessor)

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


def train(net: nn.Module, data_loader: tdata.DataLoader, optimizer: Optimizer, criterion: _Loss, data_size: int = None,
          log_batch_num: int = None):
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


def test(net: nn.Module, data_loader: tdata.DataLoader, criterion, data_size: int = None):
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
    train_loader, val_loader, (train_size, val_size) = load_data(CSV_PATH, split_ratio)
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


def plot(file_path):
    with open(file_path, 'rb') as f:
        result = pickle.load(f)
    stat = result['stat']
    train_loss, train_acc, val_loss, val_acc = stat['train_loss'], stat['train_acc'], stat['val_loss'], stat['val_acc']
    x = np.arange(1, len(train_loss) + 1)

    plt.figure()
    plt.title('loss')
    plt.semilogy(x, train_loss, label='training')
    plt.semilogy(x, val_loss, label='validation')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()

    plt.figure()
    plt.title('accuracy')
    plt.plot(x, train_acc, label='training')
    plt.plot(x, val_acc, label='validation')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.show()


if __name__ == '__main__':
    main()
