import glob
import os
import pickle
import pandas as pd
#from nltk.tokenize import sent_tokenize
#from nltk.tokenize import word_tokenize
import gzip
import re
import logging

logger = logging.getLogger()
# logging.basicConfig(filename='log_amazonPreprocess_Electronics.txt', level=logging.INFO, format=' %(asctime)s - %(levelname)s - %(message)s')
#logging.basicConfig(filename='log_amazonPreprocess_Books.txt', level=logging.INFO, format=' %(asctime)s - %(levelname)s - %(message)s')
#logging.basicConfig(filename='log_amazonPreprocess_Kitchen.txt', level=logging.INFO, format=' %(asctime)s - %(levelname)s - %(message)s')
n_out_f = 495
out_dir = '/home/DebanjanChaudhuri/topic_lms/data/amazon/amazon_data/book_chunks/'


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


def get_chunks(dat, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(dat), n):
        yield dat[i:i + n]


def write_chunks(data, file_n):
    print (file_n, data[0], len(data))
    word_tokenized = [[(sent + ' eos').split() for sent in review] for review in data]
    with open(out_dir + 'book_' + file_n + '.pkl', 'wb') as f:
        pickle.dump(word_tokenized, f)


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

    batch_size = int(len(tokenized) / n_out_f)
    print (batch_size)

    logger.info("word tokenization")
    i=0


    #word_tokenized = [[word_tokenize(sent) for sent in review] for review in tokenized]

    #path = '/data/user/apopkes/data/amazon/word_tokenized_'+name
    # with open(path, 'wb') as f:
    #     pickle.dump(word_tokenized, f)

    # logger.info("insert end-of-sentence markers")
    # # add end-of-sentence markers to each sentence
    # [sent.append('eos') for review in word_tokenized for sent in review]
    #
    # path = '/data/user/apopkes/data/amazon/word_tokenized_eos_'+name
    # with open(path, 'wb') as f:
    #     pickle.dump(word_tokenized, f)
    for chunk in get_chunks(tokenized, batch_size):
        write_chunks(chunk, str(i))
        i = i + 1

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




