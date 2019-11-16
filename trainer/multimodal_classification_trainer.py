import torch
from torch import nn
from torch.optim.optimizer import Optimizer
from torch.utils import data as tdata

from helper import get_accuracy
from trainer.trainer import Trainer, Tester


class MultimodalClassificationTrainer(Trainer):
    def __init__(self, net: nn.Module,
                 optimizer: Optimizer,
                 data_loader: tdata.DataLoader,
                 data_size: int = None):
        """
        Args:
            net (nn.Module): model structure
            data_loader (torch.utils.data.DataLoader): data loader
            optimizer (torch.optim.optimizer.Optimizer): optimizer
            data_size (int): total number of data, necessary when using a sampler to split training and validation data.
        """
        self.net = net
        self.optimizer = optimizer
        self.data_loader = data_loader
        self.data_size = data_size

    def __call__(self, log_batch_num: int = None):

        """Train one epoch

        Args:
            log_batch_num (int): print count, the statistics will print after given number of steps

        Returns:
            Tuple of training loss and training accuracy
        """
        self.net.train()

        losses = 0.
        accs = 0.
        running_loss = 0.
        running_accs = 0.

        for batch_num, samples in enumerate(self.data_loader, 0):
            elite = samples['elite'].cuda()
            text = samples['text'].cuda()
            labels = samples['label'].cuda()

            self.optimizer.zero_grad()

            # forward + backward + optimize
            outputs = self.net(text, elite)
            loss = nn.CrossEntropyLoss(outputs, labels)
            loss.backward()
            self.optimizer.step()
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

        dataset_size = len(self.data_loader.dataset) if not self.data_size else self.data_size
        return losses / dataset_size, accs / dataset_size


class MultimodalClassificationTester(Tester):
    def __init__(self, net: nn.Module, data_loader: tdata.DataLoader, data_size: int = None):
        super().__init__()
        self.net = net
        self.data_loader = data_loader
        self.data_size = data_size

    def __call__(self):
        """Test (validate) one epoch of the given data

        Returns:
            Tuple of test loss and test accuracy
        """
        self.net.eval()

        loss = 0.
        acc = 0.

        for samples in self.data_loader:
            elite = samples['elite'].cuda()
            text = samples['text'].cuda()
            labels = samples['label'].cuda()

            # forward
            output = self.net(text, elite)
            loss += nn.CrossEntropyLoss(output, labels).item()
            acc += get_accuracy(output, labels)

        dataset_size = len(self.data_loader.dataset) if not self.data_size else self.data_size
        return loss / dataset_size, acc / dataset_size
