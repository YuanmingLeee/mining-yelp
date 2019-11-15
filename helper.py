import argparse
import pickle
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from torch import nn
from torch.nn.modules.loss import _Loss
from torch.optim.optimizer import Optimizer
from torch.utils import data as tdata

from configs import OUTPUT_DIR


def parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    parser_plot = subparsers.add_parser('plot')
    parser_plot.add_argument('file_path', type=str, help='File path to the saved statistic file')
    parser_plot.set_defaults(func=plot)

    return parser.parse_args()


def get_accuracy(scores: torch.Tensor, labels: torch.Tensor):
    predicted_labels = scores.argmax(dim=1)
    indicator = (predicted_labels == labels)
    num_matches = indicator.sum()

    return num_matches.float()


def plot(args):
    """Plot loss, accuracy over epochs for training and validation. Two graphs will popup.

    Args:
        args: An argument list containing:
            file_path (str): Pickle file created by train_elite.py. Please refer to train_elite.py for
                object structure.
    """
    file_path = args.file_path
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


def train_one_epoch(net: nn.Module,
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
        elite = samples['elite'].cuda()
        text = samples['text'].cuda()
        labels = samples['label'].cuda()

        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(text, elite)
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


def test_one_epoch(net: nn.Module,
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
        elite = samples['elite'].cuda()
        text = samples['text'].cuda()
        labels = samples['label'].cuda()

        # forward
        output = net(text, elite)
        loss += criterion(output, labels).item()
        acc += get_accuracy(output, labels)

    dataset_size = len(data_loader.dataset) if not data_size else data_size
    return loss / dataset_size, acc / dataset_size


def train_net(net: nn.Module,
              epochs: int,
              optimizer: Optimizer,
              criterion: _Loss,
              train_loader: tdata.DataLoader,
              val_loader: tdata.DataLoader,
              save_name: str = '',
              train_size: int = None,
              val_size: int = None):
    """
    Train model general utility functions

    Args:
        net (nn.Module):
        epochs (int):
        optimizer (Optimizer):
        criterion (_Loss):
        train_loader (tdata.DataLoader):
        val_loader (tdata.DataLoader):
        save_name (str):
        train_size (int):
        val_size (int): Optional. Need to be set when a data loader is split by a sampler
    Return:
        A dictionary containing statistic results: training and validation loss and accuracy, and training
            parameters.
    """
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
        train_loss, train_acc = train_one_epoch(net, data_loader=train_loader, optimizer=optimizer, criterion=criterion,
                                                data_size=train_size)
        val_loss, val_acc = test_one_epoch(net, data_loader=val_loader, criterion=criterion, data_size=val_size)

        # process statistics
        train_losses[epoch], train_accs[epoch] = train_loss, train_acc
        val_losses[epoch], val_accs[epoch] = val_loss, val_acc

        t.set_description('[epoch {:d}] train loss {:g} | acc {:g} || val loss {:g} | acc {:g}'
                          .format(epoch, train_loss, train_acc, val_loss, val_acc))

        # save model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(net.state_dict(), OUTPUT_DIR / '{}-{:s}.pth'.format(save_name, name_seed))

    stat = {'train_loss': train_losses, 'train_acc': train_accs, 'val_loss': val_losses, 'val_acc': val_accs}

    return {'stat': stat, 'info': {'name_seed': name_seed}}


if __name__ == '__main__':
    args_ = parse_args()
    args_.func(args_)
