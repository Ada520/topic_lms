import os
import pickle
import random
import ipdb
import numpy as np
import random
from nltk.corpus import stopwords
import logging
import glob
logger = logging.getLogger()


logging.basicConfig(level=logging.INFO, format=' %(asctime)s - %(levelname)s - %(message)s')
#logging.basicConfig(filename='log_amazon.txt', level=logging.INFO, format=' %(asctime)s - %(levelname)s - %(message)s')


def random_reviews(path, number):
    """
    Randomly selects a given number of reviews and
    saves the resulting list to disk.
    """
    with open(path, 'rb') as f:
        reviews = pickle.load(f)

    random_idx = np.random.randint(low=0, high=len(reviews), size=number)

    ipdb.set_trace()
    subset_reviews = [reviews[i] for i in random_idx]

    data_path = '/data/user/apopkes/data/amazon/random_reviews/reviews_'
    name = path.split('_')[-1]
    save_path = data_path+name
    with open(save_path, 'wb') as f:
        pickle.dump(subset_reviews, f)

def create_dataset():
    logger.info("Step 1: load all files")
    data_path = '/data/user/apopkes/data/amazon/random_reviews/reviews_'
    with open(data_path+'Electronics', 'rb') as f:
        data_electro = pickle.load(f)
    with open(data_path+'Books', 'rb') as f:
        data_books = pickle.load(f)
    with open(data_path+'Kitchen', 'rb') as f:
        data_kitchen = pickle.load(f)
    with open(data_path+'TV', 'rb') as f:
        data_tv = pickle.load(f)

    logger.info("Create category lists")
    # Step 2: For each review create a list that holds the category name for each word in each review
    electro = [[['electro' for word in sentence] for sentence in review] for review in data_electro]
    books = [[['books' for word in sentence] for sentence in review] for review in data_books]
    kitchen = [[['kitchen' for word in sentence] for sentence in review] for review in data_kitchen]
    tv = [[['tv' for word in sentence] for sentence in review] for review in data_tv]

    logger.info("Step 3: Join review and category lists")
    # Step 3: Join lists of all categories and their keyword lists
    all_data = data_kitchen + data_books + data_electro + data_tv
    all_categories = kitchen + books + electro + tv

    logger.info("Step 4: Shuffle lists")
    # Step 4: Shuffle both lists in the same way
    joined = list(zip(all_data, all_categories))
    random.shuffle(joined)
    all_data, all_categories = zip(*joined)

    all_data = list(all_data)
    all_categories = list(all_categories)

    logger.info("Save dataset and categories to disk")
    save_data = '/data/user/apopkes/data/amazon/mixed_dataset'
    save_categories = '/data/user/apopkes/data/amazon/mixed_dataset_categories'

    with open(save_data, 'wb') as f:
        pickle.dump(all_data, f)
    with open(save_categories, 'wb') as f:
        pickle.dump(all_categories, f)

    ipdb.set_trace()
    logger.info("Step 5: split into training and test set")
    # Step 5: Split the dataset into a training and test set
    number_reviews = len(all_data)
    split = int(number_reviews/100 * 80)

    train_set = all_data[:split]
    train_categories = all_categories[:split]
    logger.info("Number of training reviews: {}".format(len(train_set)))
    train_d = '/data/user/apopkes/data/amazon/train_data'
    train_c = '/data/user/apopkes/data/amazon/train_categories'

    with open(train_d, 'wb') as f:
        pickle.dump(train_set, f)

    with open(train_c, 'wb') as f:
        pickle.dump(train_categories, f)

    test_set = all_data[split:]
    test_categories = all_categories[split:]
    logger.info("Number of test reviews: {}".format(len(test_set)))
    test_d = '/data/user/apopkes/data/amazon/test_data'
    test_c = '/data/user/apopkes/data/amazon/test_categories'

    with open(test_d, 'wb') as f:
        pickle.dump(test_set, f)

    with open(test_c, 'wb') as f:
        pickle.dump(test_categories, f)

def create_lda_dataset():
    ipdb.set_trace()
    logger.info("Create LDA dataset")
    train_d = '/data/user/apopkes/data/amazon/train_data'
    with open(train_d, 'rb') as f:
        data = pickle.load(f)

    sent_list = [sent for review in data for sent in review]
    no_stopwords = [[word for word in sent if word not in stopwords.words('english')] for sent in sent_list]
    path = '/data/user/apopkes/data/amazon/lda_train'

    with open(path, 'wb') as f:
        pickle.dump(no_stopwords, f)

    # logger.info("Flatten lists")
    # # flatten each review to get a list of reviews
    # flattened = []
    # for review in word_tokenized:
    #     flatten = [word for sent in review for word in sent]
    #     flattened.append(flatten)

    # path = '/data/user/apopkes/data/amazon/flattened_'+name
    # with open(path, 'wb') as f:
    #     pickle.dump(flattened, f)




if __name__=="__main__":
    # file_list = glob.glob('/data/user/apopkes/data/amazon/word_tokenized_eos_*')
    # for filepath in file_list:
    #     random_reviews(filepath, number=6000)

    ipdb.set_trace()
    create_dataset()
    # create_lda_dataset()

