import glob
import os
import pickle
import pandas as pd
#from nltk.tokenize import sent_tokenize
#from nltk.tokenize import word_tokenize
import gzip
#import ipdb
import re
import regex
#from nltk.corpus import stopwords
import logging

logger = logging.getLogger()
# logging.basicConfig(filename='log_amazonPreprocess_Electronics.txt', level=logging.INFO, format=' %(asctime)s - %(levelname)s - %(message)s')
#logging.basicConfig(filename='log_amazonPreprocess_Books.txt', level=logging.INFO, format=' %(asctime)s - %(levelname)s - %(message)s')
#logging.basicConfig(filename='log_amazonPreprocess_Kitchen.txt', level=logging.INFO, format=' %(asctime)s - %(levelname)s - %(message)s')

def parse(path):
    g = gzip.open(path, 'rb')
    for l in g:
        yield eval(l)

def getDF(path):
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')

def preprocess_reviews(path, name):
    # logger.info("get dataframge")
    # df = getDF(path)
    # reviews = df['reviewText'].tolist()
    # processed_reviews = []

    # logger.info("basic preprocessing")
    # for item in reviews:
    #     temp = item.lower()
    #     # Replace all tokens that contain numbers with the string "number"
    #     temp = re.sub("\S*\d\S*", "number", temp)
    #     # Remove new lines
    #     temp = re.sub('\r+\n*', " ", temp)
    #     temp = re.sub('\t', ' ', temp)
    #     # Remove text within brackets
    #     temp = re.sub("[\(\[].*?[\)\]]", "", temp)
    #     # Remove multiple spaces between words
    #     temp = re.sub(' +',' ', temp)
    #     # Remove duplicated words in row
    #     temp = re.sub(r'\b(\w+)( \1\b)+', r'\1', temp)

    #     processed_reviews.append(temp)

    # logger.info("tokenize step 1")
    # tokenized_temp = [sent_tokenize(review) for review in processed_reviews]
    #path = '/data/user/apopkes/data/amazon/temp_tokenized_'+name
    # with open(path, 'wb') as f:
    #     pickle.dump(tokenized_temp, f)
    #with open(path, 'rb') as f:
    #    tokenized_temp = pickle.load(f)

    #logger.info("restrict dataset size to 500.000 reviews")
    #tokenized_temp = tokenized_temp[:500000]

    #logger.info("tokenize step 2")
    #tokenized = [[re.sub('[^`A-Za-z0-9ÄÜÖßäüö ]+', '', sent) for sent in review] for review in tokenized_temp]
    #path = '/data/user/apopkes/data/amazon/tokenized_'+name
    #with open(path, 'wb') as f:
    #    pickle.dump(tokenized, f)

    with open(path, 'rb') as f:
        tokenized = pickle.load(f)

    logger.info("word tokenization")
    word_tokenized = [[(sent+' eos').split() for sent in review] for review in tokenized]

    path = '/data/user/apopkes/data/amazon/word_tokenized_'+name
    with open(path, 'wb') as f:
        pickle.dump(word_tokenized, f)

    logger.info("insert end-of-sentence markers")
    # add end-of-sentence markers to each sentence
    #[sent.append('eos') for review in word_tokenized for sent in review]

    path = '/data/user/apopkes/data/amazon/word_tokenized_eos_'+name
    with open(path, 'wb') as f:
        pickle.dump(word_tokenized, f)

    logger.info("End of program")

if __name__ == "__main__":

    #file_list = glob.glob('/data/user/apopkes/data/amazon/raw/*')
    #file_list = glob.glob('/data/user/apopkes/data/amazon/raw/reviews_Home_and_Kitchen.json.gz')
    file_list = glob.glob('/home/DebanjanChaudhuri/topic_lms/data/amazon/amazon_data/temp_tokenized_Books')
    #file_list = glob.glob('/data/user/apopkes/data/amazon/tokenized_Kitchen')
    # file_list = glob.glob('/data/user/apopkes/data/amazon/tokenized_Electronics')

    for path in file_list:
        name = path.split('_')[-1].split('.')[0]
        preprocess_reviews(path, name)




