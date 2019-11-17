from string import punctuation
from collections import Counter
from datetime import datetime as dt
import nltk
from nltk.corpus import stopwords
from os.path import abspath, join, exists
from os import makedirs
import pandas as pd
import json
import csv
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from random import sample
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
import pickle

def review_preprocessing(merged_review_csv_dir, glove_embedding_dir, output_dir, train_ratio=0.9, keep_ratio=0.2, review_length_max=300, review_length_min=10, seq_length=200, embedding_length=50):
    # split data
    train , test = split_dataset(merged_review_csv_dir, train_ratio)
    print("---- number of training data : {}, number of testing data : {}".format(len(train), len(test)))

    # tokenize and create word2int mapping
    print("---- tokenize reviews, remove punctuation and stop words")
    stop_words = set(stopwords.words('english'))
    all_words = []

    for index, line in tqdm(train.iterrows(), total=len(train)):
        tokens = nltk.word_tokenize(line[0])
        tokens = [word.lower() for word in tokens if word not in punctuation and word not in stop_words]
        all_words += tokens
    
    # create word to int mapping
    print("---- create word2int mapping")
    mapping = create_mapping(all_words, keep_ratio)
    with open(join(output_dir, "mapping.pickle"), 'wb') as f:
        pickle.dump(mapping, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("---- word to int mapping is saved")

    vocab_size = len(mapping) + 1
    create_pretrained_weights(glove_dir = glove_embedding_dir, word_to_int = mapping, output_dir = output_dir, vocab_size = vocab_size, embedding_length = embedding_length)

    # process both training and testing dataset and save as numpy array
    print("---- converting sentence to vector")
    train_x = []
    train_y = []
    test_x = []
    test_y = []

    print("---- processing the training dataset")
    for index, row in tqdm(train.iterrows(), total=len(train)):
        tokens = nltk.word_tokenize(row[0])
        tokens = [word.lower() for word in tokens if word not in punctuation and word not in stop_words]
        int_vec = map_sentence_to_int(tokens, mapping)
        
        if len(int_vec) > review_length_max or len(int_vec) < review_length_min:
            continue

        int_seq = truncate_or_padding(int_vec, seq_length)
        label = [int(row[1])]
        train_x.append(int_seq)
        train_y.append(label)

    print("---- processing the test dataset")
    for index, row in tqdm(test.iterrows(), total=len(test)):
        tokens = nltk.word_tokenize(row[0])
        tokens = [word.lower() for word in tokens if word not in punctuation and word not in stop_words]
        int_vec = map_sentence_to_int(tokens, mapping)
        
        if len(int_vec) > review_length_max or len(int_vec) < review_length_min:
            continue

        int_seq = truncate_or_padding(int_vec, seq_length)
        label = [int(row[1])]
        test_x.append(int_seq)
        test_y.append(label)

    train_x = np.array(train_x)
    train_y = np.array(train_y)
    test_x = np.array(test_x)
    test_y = np.array(test_y)

    np.save(join(output_dir, "text_lstm_train_x.npy"), train_x)
    np.save(join(output_dir, "text_lstm_train_y.npy"), train_y)
    np.save(join(output_dir, "text_lstm_test_x.npy"), test_x)
    np.save(join(output_dir, "text_lstm_test_y.npy"), test_y)
    
    print("---- all dataset saved successfully!")


def create_mapping(words, keep_ratio):
    counts = Counter(words)
    freq = sorted(counts.values())
    total_num = len(freq)
    threshold_index = int(total_num * (1 - keep_ratio))
    sorted_counts = counts.most_common( int(keep_ratio * total_num) )
    vocab_mapping = { w: i+1 for i, (w, c) in enumerate(sorted_counts)}
    vocab_mapping['unk'] = len(vocab_mapping) + 1
    print("---- frequency for threshold : {}".format(freq[threshold_index]))
    print("---- vocab size : {}".format(len(vocab_mapping)))
    return vocab_mapping


def map_sentence_to_int(word_list, mapping):
    res = []
    for word in word_list:
        if word in mapping:
            res.append(mapping[word])
        else:
            res.append(mapping['unk'])
    return res


def create_pretrained_weights(glove_dir, word_to_int, output_dir, vocab_size, embedding_length):
    # load glove vectors and convert and save into tmp_file
    glove_file = datapath(glove_dir)
    tmp_file = get_tmpfile("test_word2vec.txt")
    glove2word2vec(glove_file, tmp_file)

    # load vectors
    wvmodel = KeyedVectors.load_word2vec_format(tmp_file)
    
    # load weights
    print("---- creating weights from pretrained glove")
    weights = np.zeros( (vocab_size, embedding_length))
    for i in tqdm(range(len(wvmodel.index2word))):
        try:
            word = wvmodel.index2word[i]
            index = word_to_int[word]
        except:
            continue
        weights[index, :] = wvmodel.get_vector(word)
    
    save_file = join(output_dir, "pretrained_weights.npy")
    np.save(save_file, weights)

def truncate_or_padding(tokens, seq_length):
    length = len(tokens)

    if length > seq_length:
        return tokens[ : seq_length]
    else:
        zeros = list(np.zeros(seq_length - length))
        return zeros + tokens

def filter_json_preprocess(review_dir, output_dir):
    print("---- checking path")
    if not exists(output_dir):
        makedirs(output_dir)

    print("---- processing json file: {}".format(review_dir))

    # read json and convert to csv
    data = open(review_dir, 'r')
    output_f = open(join(output_dir, "filtered_data.csv"), 'w')
    output_writer = csv.writer(output_f)
    output_writer.writerow(["text", "useful"])
    data.readline()

    min_date = dt.strptime("2010-01-01", "%Y-%m-%d")
    max_date = dt.strptime("2016-12-31", "%Y-%m-%d")

    for line in data:
        line_json = json.loads(line)
        useful_score = float(line_json['useful'])

        json_date = line_json['date'].split()[0]
        json_date = dt.strptime(json_date, "%Y-%m-%d")

        if json_date < min_date or json_date > max_date:
            continue

        output_writer.writerow([line_json['text'], useful_score])

    output_f.close()
    print('---- saved to filtered_data.csv ...')

def sample_review(review_dir, output_dir, n):
    print("---- checking path")
    if not exists(output_dir):
        makedirs(output_dir)

    print("---- initialize writers")
    data = open(review_dir, 'r')
    data_read = csv.reader(data, delimiter=',')

    #output_useless = open(join(output_dir, "useless.csv"), 'w')
    output_not_useful = open(join(output_dir, "not_very_useful.csv"), 'w')
    output_very_useful = open(join(output_dir, "very_useful.csv"), 'w')
    #output_useless_writer = csv.writer(output_useless)
    output_not_useful_writer = csv.writer(output_not_useful)
    output_very_useful_writer = csv.writer(output_very_useful)

    data.readline()

    print("---- sample reviews")
    #useless = []
    not_useful = []
    very_useful = []

    for line in tqdm(data_read):
        score = float(line[1])
        text = line[0]
        # if score == 0:
        #     useless.append(text)
        if score < 10:
            not_useful.append(text)
        else:
            very_useful.append(text)

    #print("---- num of useless reviews: {}".format(len(useless)))
    print("---- num of not very useful reviews: {}".format(len(not_useful)))
    print("---- num of very usseful reviews: {}".format(len(very_useful)))
    print("---- storing sampled data")
    # sample_useless = sample(useless, n)
    # for text in tqdm(sample_useless):
    #     output_useless_writer.writerow([text, 0])
    # output_useless.close()
    
    sample_not_useful = sample(not_useful, n)
    for text in tqdm(sample_not_useful):
        output_not_useful_writer.writerow([text, 0])
    output_not_useful.close()

    sample_very_useful = sample(very_useful, n)
    for text in tqdm(sample_very_useful):
        output_very_useful_writer.writerow([text, 1])
    output_very_useful.close()

def count_review_length(sample_csv_dir):
    data = open(sample_csv_dir, 'r')
    data_reader = csv.reader(data)
    counter = 0
    counts = []
    stop_words = set(stopwords.words('english'))

    print("---- counting review length")
    for line in tqdm(data_reader):
        tokens = nltk.word_tokenize(line[0])
        tokens = [word.lower() for word in tokens if word not in punctuation and word not in stop_words]
        counts.append(len(tokens))
        counter += 1
    
    print("---- number of reviews: {}".format(counter))
    print("---- visualizing review length distribution")
    pd.Series(counts).hist(bins=100)
    plt.show()


def merge_csv(csv_dir_list, output_dir):
    result = pd.concat([pd.read_csv(df, header=None) for df in csv_dir_list], ignore_index=True)
    print("---- total number of lines: {}".format(len(result)))
    result.to_csv(join(output_dir, 'merged_data.csv'), index=False, header=None)


def split_dataset(csv_dir_list, train_ratio):
    # read data
    data = pd.read_csv(csv_dir_list, header=None, usecols=[0, 1])
    limit = int(len(data) // 2 * train_ratio)

    useless = data.loc[ data[1] == 0 ]
    useful = data.loc[ data[1] == 1 ]
    very_useful = data.loc[ data[1] == 2 ]

    train = pd.concat([
        useless[0: limit],
        useful[0: limit],
        very_useful[0: limit]
    ], ignore_index=True, sort =False)

    test = pd.concat([
        useless[limit : ],
        useful[limit : ],
        very_useful[limit : ]
    ], ignore_index=True, sort =False)

    return train , test

