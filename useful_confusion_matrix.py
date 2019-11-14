import torch
from sklearn.metrics import confusion_matrix
from torch import nn
from torch.utils import data as tdata

from configs import DATA_DIR, OUTPUT_DIR, BASE_DIR
from data_engine.data_loader import load_data, prenet_preprocessor
from data_engine.dataset import PreNetDataset
from models.EliteNet import EliteNet
from models.PreNet import PreNet
from models.TextLSTM import TextLSTM

random_seed = 42
split_ratio = .2
bs = 2048
lr = 1e-3
epochs = 100
eps = 1e-8
weight_decay = 1e-6
CSV_PATH = DATA_DIR / 'combined-usefulness.csv'
WORD_2_INDEX_MAPPING_PATH = DATA_DIR / 'mapping.pickle'


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
        elite = samples['elite'].cuda()
        text = samples['text'].cuda()
        labels = samples['label'].cuda()

        # forward
        output = net(text, elite)
        gt += labels.tolist()
        pred += torch.argmax(output, dim=1).tolist()

    dataset_size = len(data_loader.dataset) if not data_size else data_size
    cm = confusion_matrix(gt, pred, labels=[0, 1])
    print(cm)
    return loss / dataset_size, acc / dataset_size


def main():
    # prepare data
    print('Loading data...')
    dataset = PreNetDataset(CSV_PATH, preprocessor=prenet_preprocessor, word2int_mapping_path=WORD_2_INDEX_MAPPING_PATH)
    train_loader, val_loader, (train_size, val_size) = load_data(dataset, split_ratio, bs=bs)
    print('Finish loading')

    # model
    elite_net = EliteNet(BASE_DIR / 'configs/user-elite.yaml').cuda()
    elite_net.load_state_dict(torch.load(OUTPUT_DIR / 'user-elite-clean.pth'))
    test_lstm = TextLSTM(BASE_DIR / 'configs/text-lstm.yaml').cuda()
    test_lstm.load_state_dict(torch.load(OUTPUT_DIR / 'useful_pred_lstm_weights.pth'))
    net = PreNet(test_lstm, elite_net).cuda()
    net.load_state_dict(torch.load(OUTPUT_DIR / 'prenet.pth'))
    test(net, data_loader=val_loader, data_size=val_size)


if __name__ == '__main__':
    main()
