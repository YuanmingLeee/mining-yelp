import abc
from datetime import datetime
from typing import Tuple

import numpy as np
import torch
import tqdm
from torch import nn

from configs import OUTPUT_DIR


class Trainer(abc.ABC):
    """
    Trainer base class. When used train_net API, a customer trainer should extend from this class.
    """

    @abc.abstractmethod
    def __call__(self, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        pass


class Tester(abc.ABC):
    """
    Test base class. When used train_net API, a customer tester should extend from this class.
    """

    @abc.abstractmethod
    def __call__(self, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        pass


def train_net(net: nn.Module,
              epochs: int,
              trainer: Trainer,
              tester: Tester,
              save_name: str = '',
              **kwargs):
    """
    Train model general utility functions

    Args:
        net (nn.Module):
        epochs (int):
        trainer (Callable):
        tester (Callable): Model validation/test functions. It should be return
        save_name (str): Optional. Saved name of model and training statistic result.

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
        train_loss, train_acc = trainer(**kwargs)
        val_loss, val_acc = tester(**kwargs)

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
