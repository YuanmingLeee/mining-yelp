import argparse
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch


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


if __name__ == '__main__':
    args_ = parse_args()
    args_.func(args_)
