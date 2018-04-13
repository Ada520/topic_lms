import pickle
from collections import Counter
import pandas as pd
import os
import numpy as np
import logging
import ipdb

logger = logging.getLogger()
logging.basicConfig(level=logging.DEBUG, format=' %(asctime)s - %(levelname)s - %(message)s')


def get_vocabulary(pathname, save_vocab, vocab_size=None, min_count=None):
    """
    Computes a list of the most frequent words in the given file

    Args:
        pathname: string, path where the result should be saved
        filename: string, name of file with list of tokens
        vocab_size: int, size the resulting vocabulary should have
        min_count: int, the number of times each token has to show up to be included in the vocabulary

        Either vocab_size or min_count must be given!
    """
    logger.info("Compute list of the most frequent words")
    logger.info('Read in data')
    with open(pathname, "rb") as fp:
        data = list(pickle.load(fp))

    flattened = [word for review in data for sent in review for word in sent]

    logger.info("Count frequency of words in file")
    word_freq = Counter(flattened)

    # Turn Counter into Pandas Series for further analysis
    series = pd.Series(word_freq)

    if min_count is None:
        logger.info("Compute the most common words")
        # Sort the series and get the words with the largest counts
        mostCommon = series.sort_values(ascending=False)[:vocab_size]

    else:
        mostCommon = series[series.values >= min_count]

    # Convert the result to a list and save it to disk
    mostCommon = pd.Index.tolist(mostCommon.index)
    logger.info("Length of vocab: {}".format(len(mostCommon)))

    # save_vocab = '/data/user/apopkes/data/amazon/vocab'
    with open(save_vocab, 'wb') as f:
        pickle.dump(mostCommon, f)


def remove_rare(filepath, vocab_path, save_path):
    """
    Replaces all words that are not in vocab with the token "rare"

    Args:
        pathname: path were to save the result
        filename: path of file to be transformed
        vocab: name of vocabulary file
    """
    logger.info("Loading all sentences")
    with open(filepath, "rb") as fp:
         data = pickle.load(fp)

    logger.info("Loading vocabulary")
    with open(vocab_path, "rb") as fp:
         vocabulary = pickle.load(fp)

    logger.info("Transform sentences")
    sentences = [sent for review in data for sent in review]
    transformed_sentences = [[word if word in vocabulary else 'rare' for word in sentence] for sentence in sentences]

    logger.info("Flatten the list")
    transformed_sentences = [word for sentence in transformed_sentences for word in sentence]

    logger.info("Save to disk")
    # save_path = '/data/user/apopkes/data/amazon/flatTransformedTokens_test'
    with open(save_path, 'wb') as f:
        pickle.dump(transformed_sentences, f)


def build_vocab(filepath, save_path):
    with open(filepath, 'rb') as f:
        words = pickle.load(f)

    word_to_id = dict(zip(words, range(len(words))))
    with open(save_path, 'wb') as f:
        pickle.dump(word_to_id, f)


def file_to_word_ids(filepath, vocab_path, save_path):
    """
    Transforms a list of words into a list of word ID's given a vocabulary.
    The transformed list is saved to disk.

    The file needs to contain a list of tokenized sentences. Example:
        sentences = ['This', 'is', 'sentence', 'one', 'This', 'is', 'sentence', 'two']

    Args:
        filepath: string, path to text file that should be transformed
        word_to_id: dictionary mapping each word to an index

    """
    logger.info("Load file")
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    with open(vocab_path, 'rb') as f:
        word_to_id = pickle.load(f)

    logger.info("Transform the file to word id's")
    words_as_ids = [word_to_id[word] for word in data]

    logger.info("Save the transformed file to disk")
    with open(save_path, 'wb') as f:
        pickle.dump(words_as_ids, f)

def split_train(filepath, categories_path, batch_size=20, num_steps=35):
    """
    Splits a given file into training and validation set
    80% Training data
    20% validation data
    """

    logger.info("Load file")
    with open(filepath, 'rb') as f:
        data = pickle.load(f)

    with open(categories_path, 'rb') as f:
        all_c = pickle.load(f)

    flat_c = [word for review in all_c for sent in review for word in sent]

    length = len(data)
    split = int((length/100) * 80)

    x_train = data[:split]
    train_c = flat_c[:split]
    x_valid = data[split:]
    valid_c = flat_c[split:]

    logger.info("Transform training set into matrix")
    n_batches = len(x_train) // batch_size # total number of batches
    x_train_matrix = np.reshape(x_train[0: batch_size * n_batches], [batch_size, n_batches])
    x_train_c = np.reshape(train_c[0: batch_size * n_batches], [batch_size, n_batches])

    save_train_data = root_path + 'amazon_train'
    with open(save_train_data, 'wb') as f:
        pickle.dump(x_train_matrix, f)

    save_train_c = root_path + 'amazon_train_categories'
    with open(save_train_c, 'wb') as f:
        pickle.dump(x_train_c, f)

    logger.info("Transform validation set into matrix")
    n_batches = len(x_valid) // batch_size # total number of batches
    x_valid_matrix = np.reshape(x_valid[0: batch_size * n_batches], [batch_size, n_batches])
    x_valid_c = np.reshape(valid_c[0: batch_size * n_batches], [batch_size, n_batches])

    save_valid_data = root_path + 'amazon_valid'
    with open(save_valid_data, 'wb') as f:
        pickle.dump(x_valid_matrix, f)

    save_valid_c = root_path + 'amazon_valid_categories'
    with open(save_valid_c, 'wb') as f:
        pickle.dump(x_train_c, f)


def split_test(filepath, categories_path, batch_size=20, num_steps=35):
    """
    Splits a given file into training and validation set
    80% Training data
    20% validation data
    """
    ipdb.set_trace()
    logger.info("Load file")
    with open(filepath, 'rb') as f:
        data = pickle.load(f)

    with open(categories_path, 'rb') as f:
        all_c = pickle.load(f)

    flat_c = [word for review in all_c for sent in review for word in sent]

    logger.info("Transform training set into matrix")
    n_batches = len(data) // batch_size # total number of batches
    x_test_matrix = np.reshape(data[0: batch_size * n_batches], [batch_size, n_batches])
    x_test_c = np.reshape(flat_c[0: batch_size * n_batches], [batch_size, n_batches])

    save_test_data = root_path + 'amazon_test'
    with open(save_test_data, 'wb') as f:
        pickle.dump(x_test_matrix, f)

    save_test_c = root_path + 'amazon_test_categories'
    with open(save_test_c, 'wb') as f:
        pickle.dump(x_test_c, f)

if __name__=="__main__":

    train_path = os.path.expanduser('~/topic_lms/data/preprocessed/word_tokenized_eos_apnews50k_train')
    valid_path = os.path.expanduser('~/topic_lms/data/preprocessed/word_tokenized_eos_apnews50k_valid')
    test_path = os.path.expanduser('~/topic_lms/data/preprocessed/word_tokenized_eos_apnews50k_test')

    # join training and validation set
    with open(train_path, 'rb') as f:
        joined = pickle.load(f)

    with open(valid_path, 'rb') as f:
        joined.extend(pickle.load(f))

    # adapt this path
    joined_path = os.path.expanduser('~/topic_lms/data/preprocessed/joined_word_tokenized_eos_apnews')

    with open(joined_path, 'wb') as f:
        pickle.dump(joined, f)

    # adapt this path
    vocab_path = os.path.expanduser('~/topic_lms/data/preprocessed/vocab_apnews')
    # ipdb.set_trace()
    get_vocabulary(joined_path, vocab_path, min_count=10)

    # adapt this path
    save_path = os.path.expanduser('~/topic_lms/data/preprocessed/apnews_flatTransformedTokens_valid')
    remove_rare(valid_path, vocab_path, save_path)

    # adapt this path
    vocab_dict = os.path.expanduser('~/topic_lms/data/preprocessed/apnews_vocab_dict')
    build_vocab(vocab_path, vocab_dict)

    word_ids = os.path.expanduser('~/topic_lms/data/preprocessed/apnews_wordids_valid')
    file_to_word_ids(save_path, vocab_dict, word_ids)

    #split_train(filepath, categories_path)
    # split_test(filepath, categories_path)

