import argparse
import math
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from sklearn.metrics import confusion_matrix

from configs import DATA_DIR
from models.MultimodalClassifier import MultimodalClassifier

def get_accuracy(scores: torch.Tensor, labels: torch.Tensor):
    predicted_labels = scores.argmax(dim=1)
    indicator = (predicted_labels == labels)
    num_matches = indicator.sum()

    return num_matches.float()


def plot(args):
    return
    """Plot loss, accuracy over epochs for training and validation. Two graphs will popup.

    Args:
        args: An argument list containing:
            file_path (str): Pickle file created by train_elite.py. Please refer to train_elite.py for
                object structure.
    """
    # file_path = args.file_path
    # with open(file_path, 'rb') as f:
    #     result = pickle.load(f)
    # stat = result['stat']
    # train_loss, train_acc, val_loss, val_acc = stat['train_loss'], stat['train_acc'], stat['val_loss'], stat['val_acc']
    # x = np.arange(1, len(train_loss) + 1)
    #
    # plt.figure()
    # plt.title('loss')
    # plt.semilogy(x, train_loss, label='training')
    # plt.semilogy(x, val_loss, label='validation')
    # plt.legend()
    # plt.xlabel('epoch')
    # plt.ylabel('loss')
    # plt.show()
    #
    # plt.figure()
    # plt.title('accuracy')
    # plt.plot(x, train_acc, label='training')
    # plt.plot(x, val_acc, label='validation')
    # plt.xlabel('epoch')
    # plt.ylabel('accuracy')
    # plt.show()


def find_confusion_matrix(args):
    return
    """Find confusion matrix given data loader

    Args:
        args: Arguments containing:
            name (str): net work string
            bs (int): prediction batch size
            split_ratio (float): test set split ratio from the dataset
            model_weight (str): saved model path

    Returns:
        Confusion matrix of the model
    """
    # from data_engine.data_loader import load_data
    #
    # args.name = args.name.lower()
    #
    # if args.name == 'elite-net':
    #     from data_engine.dataset import EliteDataset
    #     from data_engine.data_loader import elite_preprocessor
    #     from models.EliteNet import EliteNet
    #
    #     csv_path = DATA_DIR / 'user-profiling.csv'
    #
    #     dataset = EliteDataset(csv_path, preprocessor=elite_preprocessor)
    #     net = EliteNet(args.config).cuda()
    # elif args.name == 'textlstm':
    #     # TODO
    #     return
    # elif args.name == 'multimodal-classifier':
    #     from data_engine.data_loader import multimodal_classification_preprocessor
    #     from data_engine.dataset import MultimodalClassifierDataset
    #
    #     csv_path = DATA_DIR / 'combined-usefulness.csv'
    #     word_2_index_mapping_path = DATA_DIR / 'mapping.pickle'
    #
    #     dataset = MultimodalClassifierDataset(csv_path,
    #                                           preprocessor=multimodal_classification_preprocessor,
    #                                           word2int_mapping_path=word_2_index_mapping_path)
    #     net = MultimodalClassifier(args.config)
    # else:
    #     raise ValueError('Model name argument does not supported')
    #
    # # get data loader
    # print('Test set size: {}'.format(math.ceil(len(dataset) * args.split_ratio)))
    # _, data_loader, _ = load_data(dataset, args.split_ratio, bs=args.bs)
    #
    # # obtain predictions
    # net.load_state_dict(torch.load(args.model_weight))
    #
    # net.eval()
    # labels = []
    # for batch in data_loader:
    #     labels += batch['label'].tolist()
    # pred = net.batch_predict(data_loader)
    #
    # matrix = confusion_matrix(labels, pred, labels=[0, 1])
    # print(matrix)
    #
    # return matrix

