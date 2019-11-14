import torch
from sklearn.metrics import confusion_matrix
from torch import nn
from torch.utils import data as tdata

from configs import DATA_DIR, OUTPUT_DIR, BASE_DIR
from data_engine.data_loader import elite_preprocessor, load_data
from data_engine.dataset import EliteDataset
from models.EliteNet import EliteNet

random_seed = 42
split_ratio = .2
bs = 1024
lr = 1e-3
epochs = 100
eps = 1e-8
weight_decay = 1e-6
CSV_PATH = DATA_DIR / 'user-profiling.csv'
WEIGHT_PATH = OUTPUT_DIR / 'user-elite-clean.pth'


def test(net: nn.Module,
         data_loader: tdata.DataLoader,
         data_size: int = None
         ):
    """Test (validate) one epoch of the given data

    Args:
        net (nn.Module): model structure
        data_loader (torch.utils.data.DataLoader): data loader
        data_size (int): total number of data, necessary when using a sampler to split training and validation data.

    Returns:
        Tuple of test loss and test accuracy
    """
    net.eval()

    loss = 0.
    acc = 0.
    gt = []
    pred = []

    for samples in data_loader:
        features = samples['features'].cuda()
        labels = samples['label'].cuda()

        # forward
        output = net(features)
        gt += labels.tolist()
        pred += torch.argmax(output, dim=1).tolist()

    dataset_size = len(data_loader.dataset) if not data_size else data_size
    cm = confusion_matrix(gt, pred, labels=[0, 1])
    print(cm)
    return loss / dataset_size, acc / dataset_size


def main():
    # prepare data
    print('Loading data...')
    dataset = EliteDataset(CSV_PATH, preprocessor=elite_preprocessor)
    train_loader, val_loader, (train_size, val_size) = load_data(dataset, split_ratio, bs=bs)
    print('Finish loading')

    # loss
    loss = nn.CrossEntropyLoss()

    # model
    net = EliteNet(BASE_DIR / 'configs/user-elite.yaml').cuda()
    net.load_state_dict(torch.load(WEIGHT_PATH))
    test(net, data_loader=val_loader, data_size=val_size)


if __name__ == '__main__':
    main()
