import argparse
import pickle

from torch import optim

from configs import DATA_DIR, OUTPUT_DIR, BASE_DIR
from data_engine.data_loader import elite_preprocessor, load_data
from data_engine.dataset import EliteDataset
from models.EliteNet import EliteNet
from trainer.trainer import train_net
from trainer.user_elite_trainer import UserEliteTrainer, UserEliteTester

CSV_PATH = DATA_DIR / 'user-profiling.csv'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=100, help='epoch number for training')
    parser.add_argument('--bs', type=int, default=1024, help='batch size for training and testing')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--eps', type=float, default=1e-8, help='eps')
    parser.add_argument('--weight-decay', dest='weight_decay', type=float, default=1e-6, help='weight decay rate')
    parser.add_argument('--val-train-ratio', dest='split_ratio', type=float, default=0.2,
                        help='validation set ratio to the whole dataset')
    parser.add_argument('--output', type=str, default='user-elite',
                        help='output name of model and statistic result')
    parser.add_argument('--config', type=str, default=str(BASE_DIR / 'configs/user-elite.yaml'),
                        help='template configuration')

    return parser.parse_args()


def main(args):
    # prepare data
    print('Loading data...')
    dataset = EliteDataset(CSV_PATH, preprocessor=elite_preprocessor)
    train_loader, val_loader, (train_size, val_size) = load_data(dataset, args.split_ratio, bs=args.bs)
    print('Finish loading')

    # model
    net = EliteNet(args.config).cuda()

    # optimizer
    optimizer = optim.Adam(net.parameters(), lr=args.lr, eps=args.eps, weight_decay=args.weight_decay)

    # trainer
    trainer = UserEliteTrainer(net, optimizer=optimizer, data_loader=train_loader, data_size=train_size)
    tester = UserEliteTester(net, data_loader=val_loader, data_size=val_size)

    content = train_net(net, epochs=args.epoch, trainer=trainer, tester=tester, save_name=args.output)

    # save statistic
    with open(OUTPUT_DIR / '{}-stat-{}.pkl'.format(args.output, content['info']['name_seed']), 'wb') as f:
        pickle.dump(content, f)


if __name__ == '__main__':
    main(parse_args())
