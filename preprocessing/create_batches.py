import pickle
import numpy as np
import os
import itertools


corpus = 'bnc'
train_path = os.path.expanduser('~/topic_lms/data/' + corpus + '/train_transform.pkl')

with open(train_path, 'rb') as f:
    data = pickle.load(f)


# Create a padded batch
# 1. Get 64 elements from list
subset = data[:64]
# 2. Create padding
padded_subset = np.array(list(itertools.zip_longest(*subset, fillvalue=0)))
print('padded_subset: ', padded_subset.T)
# 3. Create target batch
subset_targets = np.roll(padded_subset, -1)
subset_targets[:, -1] = 0
print('subset_targets: ', subset_targets)

