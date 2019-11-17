import argparse
import math
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn import metrics
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

from configs import DATA_DIR
from models.MultimodalClassifier import MultimodalClassifier


def parse_args():
    """Argument parser"""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    parser_plot = subparsers.add_parser('plot')
    parser_plot.add_argument('file_path', type=str, help='File path to the saved statistic file')
    parser_plot.set_defaults(func=plot)

    parser_confusion_mtx = subparsers.add_parser('confusion-mtx')
    parser_confusion_mtx.add_argument('--name', type=str, help='Case insensitive Model name, support: elite-net, '
                                                               'text-lstm, and multimodal-classifier.')
    parser_confusion_mtx.add_argument('--split-ratio', dest='split_ratio', type=float, default=1,
                                      help='test set split ratio from the whole dataset')
    parser_confusion_mtx.add_argument('--bs', type=int, default=2048, help='batch size for testing')
    parser_confusion_mtx.add_argument('--model-weight', dest='model_weight', type=str, help='Path to model weight')
    parser_confusion_mtx.add_argument('config', type=str, help='model configuration file')
    parser_confusion_mtx.set_defaults(func=find_confusion_matrix)

    parser_predict = subparsers.add_parser('pred-statistical')
    parser_predict.add_argument('file_path', type=str, help='File path to pickle of saved logistic regression, SVM or'
                                                            'XGBoost model')
    parser_predict.set_defaults(func=predict)

    parser_roc = subparsers.add_parser('plot-roc')
    parser_roc.add_argument('file_path', type=str, help='File path to pickle of saved logistic regression, SVM or'
                                                        'XGBoost model')
    parser_roc.set_defaults(func=plot_roc)

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
    train_loss, train_acc, val_loss, val_acc = stat['train_loss'], stat['train_acc'], stat['test_loss'], stat[
        'test_acc']
    x = np.arange(1, len(train_loss) + 1)

    plt.figure()
    plt.title('loss')
    plt.semilogy(x, train_loss, label='training')
    plt.semilogy(x, val_loss, label='testing')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()

    plt.figure()
    plt.title('accuracy')
    plt.plot(x, train_acc, label='training')
    plt.plot(x, val_acc, label='testing')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.show()


def find_confusion_matrix(args):
    """Find confusion matrix of deep learning model given data loader. Supported models:
        elite-net, text-lstm and multimodal-classifier.

    Args:
        args: Arguments containing:
            name (str): net work string
            bs (int): prediction batch size
            split_ratio (float): test set split ratio from the dataset
            model_weight (str): saved model path

    Returns:
        Confusion matrix of the model
    """
    from data_engine.data_loader import load_torch_data

    args.name = args.name.lower()

    if args.name == 'elite-net':
        from data_engine.dataset import EliteDataset
        from data_engine.data_loader import elite_preprocessor
        from models.EliteNet import EliteNet

        csv_path = DATA_DIR / 'user-profiling.csv'

        dataset = EliteDataset(csv_path, preprocessor=elite_preprocessor)
        net = EliteNet(args.config).cuda()
    elif args.name == 'text-lstm':
        from data_engine.data_loader import text_lstm_dataloader_factory
        from models.TextLSTM import TextLSTM
        test_x = DATA_DIR / 'text_lstm_test_x.npy'
        test_y = DATA_DIR / 'text_lstm_test_y.npy'

        data_loader, data_size = text_lstm_dataloader_factory(test_x, test_y, args.bs)
        net = TextLSTM(args.config)

    elif args.name == 'multimodal-classifier':
        from data_engine.data_loader import multimodal_classification_preprocessor
        from data_engine.dataset import MultimodalClassifierDataset

        csv_path = DATA_DIR / 'combined-usefulness.csv'
        word_2_index_mapping_path = DATA_DIR / 'mapping.pickle'

        dataset = MultimodalClassifierDataset(csv_path,
                                              preprocessor=multimodal_classification_preprocessor,
                                              word2int_mapping_path=word_2_index_mapping_path)
        net = MultimodalClassifier(args.config)
    else:
        raise ValueError('Model name argument does not supported')

    if args.name != "text-lstm":
        # get data loader
        print('Test set size: {}'.format(math.ceil(len(dataset) * args.split_ratio)))
        _, data_loader, _ = load_torch_data(dataset, args.split_ratio, bs=args.bs)

    # obtain predictions
    net.load_state_dict(torch.load(args.model_weight))
    net.cuda()

    net.eval()

    pred, labels = net.batch_predict(data_loader)

    matrix = confusion_matrix(labels, pred, labels=[0, 1])
    print(matrix)

    return matrix


def _load_statistical_learning_test_data(file_path):
    """Abstracted function for loading statistical learning test data"""

    with open(DATA_DIR / 'statistical-data-loaders.pkl', 'rb') as f:
        _, test_set = pickle.load(f)

    with open(file_path, 'rb') as f:
        net = pickle.load(f)

    return net, test_set


def predict(args):
    """
    Print prediction summary of a statistical learning model. This function supports:
        logistic regression, SVM and XGBoost
        It prints the accuracy, F1 score and confusion matrix on the processed test set loaded from
        `data/logistical-data-loaders.pkl`
    Args:
        args: An argument list containing:
            file_path (str): path to the saved data loader
    """
    net, test_set = _load_statistical_learning_test_data(args.file_path)

    test_samples = test_set['features']
    test_labels = test_set['label']

    preds = net.predict(test_samples, verbose=True)

    # get accuracy, f1 score and confusion matrix
    print('Testing accuracy {}'.format(accuracy_score(test_labels, preds)))
    print('Testing F1 score: \n{}'.format(f1_score(test_labels, preds, average='weighted')))
    print('Testing Confusion Matrix score: \n{}'.format(confusion_matrix(test_labels, preds)))


def plot_roc(args):
    """
    Plot roc graph of statistical learning models. Supported models are:
        logistic regression, SVM and XGBoost
    Args:
        args: an argument list containing:
            file_path (str): path to the saved data loader
    """
    net, test_set = _load_statistical_learning_test_data(args.file_path)

    test_samples = test_set['features']
    test_labels = test_set['label']

    # get ROC graph
    y_pred_proba = net.predict_proba(test_samples)[::, 1]
    fpr, tpr, _ = metrics.roc_curve(test_labels, y_pred_proba)
    auc = metrics.roc_auc_score(test_labels, y_pred_proba)

    plt.plot(fpr, tpr, label="data 1, auc=" + str(auc))
    plt.legend(loc=4)
    plt.show()


if __name__ == '__main__':
    args_ = parse_args()
    args_.func(args_)
