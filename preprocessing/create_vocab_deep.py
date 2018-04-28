import pickle
import ipdb
import itertools
import pandas as pd
from collections import Counter
from collections import defaultdict
import os
import numpy as np
import logging

from itertools import chain

logger = logging.getLogger()
logging.basicConfig(level=logging.DEBUG, format=' %(asctime)s - %(levelname)s - %(message)s')

#additional symbols
pad_symbol = "<pad>"
start_symbol = "<go> "
end_symbol = " <eos>"
unk_symbol = "<unk>"


def get_flattened_proc(dataset):
    """

    :param dataset:
    :return: flattened sentences for a corpus.
    """
    sentences = [(start_symbol + sent + end_symbol).split() for review in dataset for sent in review]

    return [word for sentence in sentences for word in sentence], sentences


def get_wid(word, vocab_d):
    """
    get word to id from dict vocab_d
    :param word:
    :param vocab_d:
    :return:
    """
    try:
        return vocab_d[word]
    except KeyError:
        return vocab_d[unk_symbol]


def get_sent2id(doc, vocab_dict):

    # return [get_wid(w, vocab_dict)
    #         for w in doc]

    return [[get_wid(w, vocab_dict)
            for w in sent]
            for sent in doc]


def read_dataset(filename):
    """
    reads the corpus text files.
    :param filename:
    :return: docs
    """
    with open(filename, 'r') as f:
        data = f.readlines()

    out = [[sent for sent in doc.replace('\n', '').split('\t')] for doc in data]
    return out


def write_batches(raw_data, batch_size, num_steps, save_path, threshold=30):
    """
    write batches
    :param raw_data:
    :param batch_size:
    :param num_steps:
    :param save_path:
    :return:
    """
    # Split long sentences into two parts
    temp = [[sublist[:threshold], sublist[threshold:threshold+30], sublist[threshold+30:threshold+60], sublist[threshold+60:threshold+90], sublist[threshold+90:threshold+120], sublist[threshold+120:threshold+150], sublist[threshold+150:threshold+180], sublist[threshold+180:threshold:210], sublist[threshold+210:threshold+240], sublist[threshold+240:]]
            if len(sublist) > threshold
            else [sublist]
            for sublist in raw_data]

    temp_flat = [subsublist
            for sublist in temp
            for subsublist in sublist
            if len(subsublist) > 0]

    # Pad with zeros
    padded_data = np.array(list(itertools.zip_longest(*temp_flat, fillvalue=0))).T

    # data_len = len(raw_data)
    data_len = len(temp_flat)
    batch_len = data_len // batch_size # total number of batches

    # data = np.reshape(padded_data[0: batch_size * batch_len], [batch_size, batch_len])
    data = padded_data[:batch_len]
    print (len(data))
    print (data[0])
    # Save numpy array to disk
    with open(save_path, 'wb') as f:
        pickle.dump(data, f)


def create_vocab(dataset, min_freq):
    """
    create vocab given training path
    :param train_path:
    :return:
    """
    # Step 1: map words with count < min_count to "rare"
    vocab = defaultdict(float)
    out_vocab = []

    for word in dataset:
        vocab[word] += 1.0

    for k, v in vocab.items():
        if v >= min_freq:
            out_vocab.append(k)

    ipdb.set_trace()
    # Step 2: map 0.1% of most common words to "rare"
    word_freq = Counter(out_vocab)
    series = pd.Series(word_freq)
    sorted_word_freq = series.sort_values(ascending=False)#[:vocab_size]
    n_words = len(out_vocab)
    n_most_frequent = int((n_words / 100) / 10)
    new_out_vocab = series[n_most_frequent:]
    new_out_vocab = pd.Index.tolist(new_out_vocab.index)
    logger.info(f"New vocab size: {n_words - n_most_frequent}")

    logger.info('Created vocabulary!')

    return dict(zip(new_out_vocab, range(1, len(new_out_vocab) + 1)))


def preprocess_data(corpus):
    """
    preprocess the whole corpus and add additional symbols
    :param file:
    :return:
    """
    logger.info('Processing corpus:' + corpus)
    train_path = os.path.expanduser('~/topic_lms/data/' + corpus + '/train.txt')
    valid_path = os.path.expanduser('~/topic_lms/data/' + corpus + '/valid.txt')
    test_path = os.path.expanduser('~/topic_lms/data/' + corpus + '/test.txt')

    out_train = os.path.expanduser('~/topic_lms/data/' + corpus + '/train_transform.pkl')
    out_test = os.path.expanduser('~/topic_lms/data/' + corpus + '/test_transform.pkl')
    out_valid = os.path.expanduser('~/topic_lms/data/' + corpus + '/val_transform.pkl')

    out_vocab = os.path.expanduser('~/topic_lms/data/' + corpus + '/vocab.pkl')

    # process train and get vocab
    train = read_dataset(train_path)
    train_flat, train = get_flattened_proc(train)
    vocab = create_vocab(train_flat, 10)
    vocab[unk_symbol] = len(vocab) + 1

    # write vocab into file.
    with open(out_vocab, 'wb') as f:
        pickle.dump(vocab, f)

    logger.info("Length of vocabulary:" + str(len(vocab)))
    train = get_sent2id(train, vocab)
    write_batches(train, 64, 30, out_train)

    valid = read_dataset(valid_path)
    valid_flat, valid = get_flattened_proc(valid)
    valid = get_sent2id(valid, vocab)
    write_batches(valid, 64, 30, out_valid)

    test = read_dataset(test_path)
    test = get_flattened_proc(test)
    test = get_sent2id(test, vocab)
    write_batches(test, 64, 30, out_test)


if __name__ == '__main__':
    domains = ['bnc', 'imdb', 'apnews']
    for dom in domains:
        preprocess_data(dom)
