import argparse
import pickle
from datetime import datetime

import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from configs import DATA_DIR, OUTPUT_DIR


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('name', type=str, help='case insensitive model name, one of logistic, svm, and xgboost')

    return parser.parse_args()


def main(args):
    with open(DATA_DIR / 'statistical-data-loaders.pkl', 'rb') as f:
        train_set, _ = pickle.load(f)

    args.name = args.name.lower()
    if args.name == 'logistic':
        net = LogisticRegression(n_jobs=1, C=1e5, verbose=True)
    elif args.name == 'svm':
        net = SVC(kernel='rbf', C=10, gamma=10, probability=True, verbose=True)
    elif args.name == 'xgboost':
        net = xgb.XGBClassifier(max_depth=3,
                                min_child_weight=1,
                                learning_rate=0.1,
                                n_estimators=500,
                                silent=True,
                                objective='binary:logistic',
                                gamma=0,
                                max_delta_step=0,
                                subsample=1,
                                colsample_bytree=1,
                                colsample_bylevel=1,
                                reg_alpha=0,
                                reg_lambda=0,
                                scale_pos_weight=1,
                                seed=1,
                                missing=None,
                                )
    else:
        raise ValueError('model name in --name argument is not supported')

    train_samples = train_set['features']
    train_labels = train_set['label']

    if args.name in ['svm', 'xgboost']:
        train_labels = train_labels.reshape((-1,))

    print('Training')
    net.fit(train_samples, train_labels)

    print('Saving')
    name_seed = datetime.now().strftime('%m%d-%H%M%S')
    with open(OUTPUT_DIR / '{}-{}.pkl'.format(args.name, name_seed), 'wb') as f:
        pickle.dump(net, f)


if __name__ == '__main__':
    main(parse_args())
