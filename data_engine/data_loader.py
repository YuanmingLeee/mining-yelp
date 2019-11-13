import numpy as np
import pandas as pd
import torch.utils.data as tdata
from sklearn import preprocessing
import nltk
from nltk.corpus import stopwords
from string import punctuation

def load_data(dataset: tdata.Dataset, ratio: float, bs: int):
    """Prepare data from torch dataset for training and validation.
    Args:
        dataset (torch.utils.data.Dataset): loaded dataset
        ratio (float): split ratio
        bs (int): batch size

    Returns:
        Tuple of training data loader, validation data loader and
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


def preprocessor(df: pd.DataFrame, label_name: str):
    # remove unrelated info
    df.drop(columns='user_id', inplace=True)
    # split by label and balance
    positive_df = df.loc[getattr(df, label_name) == 1]
    negative_df = df.loc[getattr(df, label_name) == 0].sample(positive_df.shape[0])

    # concatenate and shuffle
    result = pd.concat([positive_df, negative_df]).sample(frac=1)

    # clean
    del positive_df, negative_df

    # min max scaler
    scaler = preprocessing.MinMaxScaler()
    return pd.DataFrame(scaler.fit_transform(result))


def elite_preprocessor(df: pd.DataFrame):
    return preprocessor(df, 'elite')


def prenet_preprocessor(df: pd.DataFrame):
    return preprocessor(df, 'usefulness')

def map_sentence_to_int(word_list, mapping):
    res = []
    for word in word_list:
        if word in mapping:
            res.append(mapping[word])
        else:
            res.append(mapping['unk'])
    return res

def text_preprocessor(df: pd.DataFrame, word2int_mapping):
    for index, row in df.iterrows():
        text = row['text']
        tokens = nltk.word_tokenize(line[0])
        tokens = [word.lower() for word in tokens if word not in punctuation and word not in stop_words]
        int_vec = map_sentence_to_int(tokens, word2int_mapping)
        
        if len(int_vec) > 2000:
            int_vec = int_vec[ : 200]
        else:
            int_vec = list(np.zeros(200 - len(int_vec))) + int_vec

        df.iloc[[index]]['text'] = int_vec

    return df[['text', 'usefulness']]


