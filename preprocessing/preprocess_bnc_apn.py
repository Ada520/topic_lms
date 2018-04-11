import pickle
from nltk import word_tokenize
import re
import ipdb
import os
import glob

filepaths = glob.glob(os.path.expanduser('~/topic_lms/data/apnews/apnews50k_valid.txt'))

for fpath in filepaths:
    name = fpath.split(sep='/')[-1].split(sep='.')[-2]
    with open(fpath, 'r') as f:
        data = f.readlines()
        processed = []
        for report in data:
            report = report.lower()
            # Replace all tokens that contain numbers with the string "number"
            report = re.sub("\S*\d\S*", "number", report)
            # Remove multiple spaces between words
            report = re.sub(' +',' ', report)
            # Remove duplicated words in row
            report = re.sub(r'\b(\w+)( \1\b)+', r'\1', report)
            processed.append(report)

        # tokenize each report into a list of sentences
        processed = [report.split(sep='\t') for report in processed]

        # remove all punctuation
        processed = [[re.sub('[^`A-Za-z0-9ÄÜÖßäüö ]+', '', sent) for sent in review] for review in processed]

        # tokenize sentences into words
        word_tokenized = [[word_tokenize(sent) for sent in review] for review in processed]

        # add end-of-sentence markers to each sentence
        [sent.append('eos') for review in word_tokenized for sent in review]
        ipdb.set_trace()

        path = os.path.expanduser('~/topic_lms/data/preprocessed/word_tokenized_eos_' + name)
        with open(path, 'wb') as f:
            pickle.dump(word_tokenized, f)

