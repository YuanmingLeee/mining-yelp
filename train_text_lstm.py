import argparse
import pickle

from torch import optim

from configs import DATA_DIR, OUTPUT_DIR, BASE_DIR
from data_engine.data_loader import text_lstm_dataloader_factory
from models.TextLSTM import TextLSTM
from trainer.TextLSTM_trainer import TextLstmTrainer, TextLstmTester
from trainer.trainer import train_net

TRAIN_X = DATA_DIR / "text_lstm_train_x.npy"
TRAIN_Y = DATA_DIR / "text_lstm_train_y.npy"
TEST_X = DATA_DIR / "text_lstm_test_x.npy"
TEST_Y = DATA_DIR / "text_lstm_test_y.npy"
MAPPING = DATA_DIR / "text_lstm_mapping.pickle"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=100, help='epoch number for training')
    parser.add_argument('--bs', type=int, default=64, help='batch size for training and testing')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--eps', type=float, default=1e-8, help='eps')
    parser.add_argument('--weight-decay', dest='weight_decay', type=float, default=1e-6, help='weight decay rate')
    parser.add_argument('--val-train-ratio', dest='split_ratio', type=float, default=0.1,
                        help='validation set ratio to the whole dataset')
    parser.add_argument('--output', type=str, default='text-lstm',
                        help='output name of model and statistic result')
    parser.add_argument('--config', type=str, default=str(BASE_DIR / 'configs/text-lstm.yaml'),
                        help='template configuration')

    return parser.parse_args()


def main(args):
    # prepare data
    print('Loading data...')
    train_loader, train_size = text_lstm_dataloader_factory(TRAIN_X, TRAIN_Y, args.bs)
    test_loader, test_size = text_lstm_dataloader_factory(TEST_X, TEST_Y, args.bs)

    print('Finish loading')

    # model
    net = TextLSTM(args.config).cuda()

    # optimizer
    optimizer = optim.Adam(net.parameters(), lr=args.lr, eps=args.eps, weight_decay=args.weight_decay)

    # trainer
    trainer = TextLstmTrainer(net, optimizer=optimizer, data_loader=train_loader, data_size=train_size)
    tester = TextLstmTester(net, data_loader=test_loader, data_size=test_size)

    content = train_net(net, epochs=args.epoch, trainer=trainer, tester=tester, save_name=args.output)

    # save statistic
    with open(OUTPUT_DIR / '{}-stat-{}.pkl'.format(args.output, content['info']['name_seed']), 'wb') as f:
        pickle.dump(content, f)


if __name__ == '__main__':
    main(parse_args())
