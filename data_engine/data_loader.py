import pickle
from string import punctuation

import nltk
import numpy as np
import pandas as pd
import torch
import torch.utils.data as tdata
from gensim.models import Doc2Vec
from nltk.corpus import stopwords
from sklearn import preprocessing
from tqdm import tqdm


def load_torch_data(dataset: tdata.Dataset, ratio: float, bs: int):
    """Prepare data from torch dataset for training and validation.
    Args:
        dataset (torch.utils.data.Dataset): loaded dataset
        ratio (float): split ratio
        bs (int): batch size

    Returns:
        A tuple of training data loader, validation data loader and
            a tuple of size containing training dataset size and validation
            dataset size respectively
    """
    dataset_size = len(dataset)

    # prepare for shuffle
    indices = np.arange(dataset_size)
    np.random.shuffle(indices)
    split_idx = int(np.floor(ratio * dataset_size))
    train_indices, val_indices = indices[split_idx:], indices[:split_idx]

    # split dataset
    train_sampler = tdata.SubsetRandomSampler(train_indices)
    val_sampler = tdata.SubsetRandomSampler(val_indices)
    train_loader = tdata.DataLoader(dataset, batch_size=bs, sampler=train_sampler)
    val_loader = tdata.DataLoader(dataset, batch_size=bs, sampler=val_sampler)

    return train_loader, val_loader, (len(train_indices), len(val_indices))


def load_statistical_learning_data(path, model: Doc2Vec):
    """
    Load data for statistical learning
    Args:
        path (str): path to tagged dataset
        model (Doc2Vec): gensim doc2vec pretrained model

    Return:
        A tuple of train set and test set
    """
    # load dataset
    with open(path, 'rb') as f:
        train_tagged, test_tagged = pickle.load(f)

    print('Preprocess training data')
    train_data = tagged_data_preprocessor(model, train_tagged)
    print('Preprocess testing data')
    test_data = tagged_data_preprocessor(model, test_tagged)

    return train_data, test_data


def tagged_data_preprocessor(model, tagged_doc):
    """
    Preprocess tagged doc

    Args:
        model: pretrained doc2vec model
        tagged_doc: tagged document dataset

    Return:
         Dictionary containing features and labels
    """
    sents = tagged_doc.values
    labels, features = zip(*[(doc.tags[0], model.infer_vector(doc.words, steps=20)) for doc in tqdm(sents)])
    features = np.asarray(features)
    labels = np.asarray(list(map(int, labels))).reshape(-1, 1)

    return {'features': features, 'label': labels}


def elite_preprocessor(df: pd.DataFrame):
    """Elite net data preprocessor

    Args:
        df (pd.DataFrame): input data frame
    Return:
        Processed data frame
    """
    # remove unrelated info
    df.drop(columns='user_id', inplace=True)
    # split by label and balance
    positive_df = df.loc[df.elite == 1]
    negative_df = df.loc[df.elite == 0].sample(positive_df.shape[0])

    # concatenate and shuffle
    result = pd.concat([positive_df, negative_df]).sample(frac=1)

    # clean
    del positive_df, negative_df

    # min max scaler
    scaler = preprocessing.MinMaxScaler()
    return pd.DataFrame(scaler.fit_transform(result))


def multimodal_classification_preprocessor(df: pd.DataFrame):
    """
    Multimodal classifier data preprocessor

    Args:
        df (pd.DataFrame): input data frame read from pandas
    Return:
        Processed data frame
    """
    # remove unrelated info
    df.drop(columns='user_id', inplace=True)
    # split by label and balance
    positive_df = df.loc[df.usefulness == 1]
    negative_df = df.loc[df.usefulness == 0].sample(positive_df.shape[0])

    # concatenate and shuffle
    result = pd.concat([positive_df, negative_df]).sample(frac=1)

    # clean
    del positive_df, negative_df

    # min max scaler
    scaler = preprocessing.MinMaxScaler()
    result.iloc[:, :13] = scaler.fit_transform(result.iloc[:, :13])
    return result


def map_sentence_to_int(word_list, mapping):
    res = []
    for word in word_list:
        if word in mapping:
            res.append(mapping[word])
        else:
            res.append(mapping['unk'])
    return res


def text_preprocessor(df: pd.DataFrame, word2int_mapping):
    def text2int_vec(text: str):
        tokens = nltk.word_tokenize(text)
        tokens = list(filter(lambda x: x not in punctuation and x not in stop_words, map(str.lower, tokens)))
        int_vec = map_sentence_to_int(tokens, word2int_mapping)

        if len(int_vec) > 200:
            int_vec = int_vec[: 200]
        else:
            int_vec = list(np.zeros(200 - len(int_vec))) + int_vec
        return pd.Series(int_vec)

    stop_words = set(stopwords.words('english'))
    vectors = df.text.apply(text2int_vec)

    return np.concatenate(
        (vectors.values, df.usefulness.values.reshape(-1, 1)),
        axis=1
    )


def text_lstm_dataloader_factory(x_dir: str, y_dir: str, bs: int):
    """
    Data loader factory class for text LSTM

    Args:
        x_dir (str): directory of features
        y_dir (str): directory of labels
        bs (int): batch size
    Return:
        Tuple of data loader and dataset size
    """
    # load data
    x = np.load(x_dir)
    y = np.load(y_dir).squeeze(1)
    size = x.shape[0]

    # create Tensor datasets
    dataset = tdata.TensorDataset(torch.from_numpy(x).to(torch.int64),
                                  torch.from_numpy(y).to(torch.long))

    # make sure to SHUFFLE your data
    dataloader = tdata.DataLoader(dataset, shuffle=True, batch_size=bs)
    return dataloader, size
