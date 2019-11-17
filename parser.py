from helper import plot, find_confusion_matrix
import argparse
import yaml

def parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    parser_plot = subparsers.add_parser('plot')
    parser_plot.add_argument('file_path', type=str, help='File path to the saved statistic file')
    parser_plot.set_defaults(func=plot)

    parser_confusion_mtx = subparsers.add_parser('confusion-mtx')
    parser_confusion_mtx.add_argument('--name', type=str, help='Case insensitive Model name, support: elite-net, '
                                                               'textlstm, and multimodal-classifier.')
    parser_confusion_mtx.add_argument('--split-ratio', dest='split_ratio', type=float, default=1,
                                      help='test set split ratio from the whole dataset')
    parser_confusion_mtx.add_argument('--bs', type=int, default=2048, help='batch size for testing')
    parser_confusion_mtx.add_argument('--model-weight', dest='model_weight', type=str, help='Path to model weight')
    parser_confusion_mtx.add_argument('config', type=str, help='model configuration file')
    parser_confusion_mtx.set_defaults(func=find_confusion_matrix)

    return parser.parse_args()


def parse_config(path):
    class Struct(object):
        def __init__(self, di):
            for a, b in di.items():
                if isinstance(b, (list, tuple)):
                    setattr(self, a, [Struct(x) if isinstance(x, dict) else x for x in b])
                else:
                    setattr(self, a, Struct(b) if isinstance(b, dict) else b)

    with open(path, 'r') as stream:
        try:
            return Struct(yaml.safe_load(stream))
        except yaml.YAMLError as exc:
            print(exc)