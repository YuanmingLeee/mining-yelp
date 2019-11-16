import argparse
import pickle

from torch import optim

from configs import DATA_DIR, BASE_DIR, OUTPUT_DIR
from data_engine.data_loader import load_data, prenet_preprocessor
from data_engine.dataset import PreNetDataset
from models.PreNet import PreNet
from trainer.multimodal_classification_trainer import MultimodalClassificationTrainer, MultimodalClassificationTester
from trainer.trainer import train_net

CSV_PATH = DATA_DIR / 'combined-usefulness.csv'
WORD_2_INDEX_MAPPING_PATH = DATA_DIR / 'mapping.pickle'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=100, help='epoch number for training')
    parser.add_argument('--bs', type=int, default=64, help='batch size for training and testing')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--eps', type=float, default=1e-8, help='eps')
    parser.add_argument('--weight-decay', dest='weight_decay', type=float, default=1e-6, help='weight decay rate')
    parser.add_argument('--val-train-ratio', dest='split_ratio', type=float, default=0.2,
                        help='validation set ratio to the whole dataset')
    parser.add_argument('--output', type=str, default='multimodal-classification',
                        help='output name of model and statistic result')
    parser.add_argument('--config', type=str, default=str(BASE_DIR / 'configs/multimodal-classification.yaml'),
                        help='template configuration')

    return parser.parse_args()


def main(args):
    # prepare data
    print('Loading data...')
    dataset = PreNetDataset(CSV_PATH, preprocessor=prenet_preprocessor, word2int_mapping_path=WORD_2_INDEX_MAPPING_PATH)
    train_loader, val_loader, (train_size, val_size) = load_data(dataset, args.split_ratio, bs=args.bs)
    print('Finish loading')

    # model
    net = PreNet(args.config).cuda()

    # optimizer
    optimizer = optim.Adam(net.parameters(), lr=args.lr, eps=args.eps, weight_decay=args.weight_decay)

    # trainer
    trainer = MultimodalClassificationTrainer(net, optimizer=optimizer, data_loader=train_loader, data_size=train_size)
    tester = MultimodalClassificationTester(net, data_loader=val_loader, data_size=val_size)

    content = train_net(net, epochs=args.epoch, trainer=trainer, tester=tester, save_name=args.output)

    with open(OUTPUT_DIR / '{}-stat-{}.pkl'.format(args.output, content['info']['name_seed']), 'wb') as f:
        pickle.dump(content, f)


if __name__ == '__main__':
    main(parse_args())
