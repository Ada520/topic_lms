import pickle
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
end_symbol = "<eos>"
unk_symbol = "<unk>"


def get_flattened_proc(dataset):
    """

    :param dataset:
    :return: flattened sentences for a corpus.
    """
    sentences = [(start_symbol + sent.replace('\'', '') + end_symbol) for review in dataset for sent in review]

    return sentences


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

    return [[get_wid(w, vocab_dict) for w in sent] for sent in doc]


def read_dataset(filename):
    """
    reads the corpus text files.
    :param filename:
    :return: docs
    """
    with open(filename, 'r') as f:
        data = f.readlines()

    out = [[sent for sent in doc.split('\n')[0].split('\t')] for doc in data]
    return out


def write_batches(raw_data, batch_size, num_steps, save_path):
    """
    write batches
    :param raw_data:
    :param batch_size:
    :param num_steps:
    :param save_path:
    :return:
    """
    data_len = len(raw_data)
    batch_len = data_len // batch_size # total number of batches

    data = np.reshape(raw_data[0: batch_size * batch_len], [batch_size, batch_len])
    data = [np.array([d for ds in dat for d in ds]) for dat in data]
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
    vocab = defaultdict(float)
    out_vocab = []
    #get word frequencies
    for sent in dataset:
        for word in sent.split():
            vocab[word] += 1.0

    for k, v in vocab.items():
        if v > min_freq:
            out_vocab.append(k)

    logger.info('Created vocabulary!')

    return dict(zip(out_vocab, range(len(out_vocab))))


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
    train = get_flattened_proc(train)
    vocab = create_vocab(train, 10)
    vocab[unk_symbol] = len(vocab) + 1
    #write vocab into file.
    with open(out_vocab, 'wb') as f:
        pickle.dump(vocab, f)
    logger.info("Length of vocabulary:" + str(len(vocab)))
    train = get_sent2id(train, vocab)
    print (train[0])
    write_batches(train, 64, 30, out_train)

    # write valid
    valid = read_dataset(valid_path)
    valid = get_flattened_proc(valid)
    valid = get_sent2id(valid, vocab)
    print (valid[0])
    write_batches(valid, 64, 30, out_valid)

    # write test

    # write valid
    test = read_dataset(test_path)
    test = get_flattened_proc(test)
    test = get_sent2id(test, vocab)
    print (test[0])
    write_batches(test, 64, 30, out_test)


if __name__ == '__main__':
    domains = ['bnc', 'imdb', 'apnews']
    for dom in domains:
        preprocess_data(dom)