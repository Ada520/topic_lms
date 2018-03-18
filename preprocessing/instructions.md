TO DO 

General note: The files called "word_tokenized_eos_TV" etc. are the files you need, the other ones are just backups from intermediate preprocessing steps

1. Finish preprocessing the books dataset
- see script "preprocess reviews"
- the first two steps are done (tokenize step 1, tokenize step 2)
- still missing are "word tokenization" and "insert end-of-sentence markers"
- the relevant parts of the code are uncommented
- you just have to specify the correct path to the file "temp_tokenized_Books" file in the "if __name__ == ___main___" part and continue running the preprocessing
- note: you also have to adapt the paths where the created files will be saved

2. Mix the reviews
3. Split them into training, validation and test set

Both steps are in the "mix_reviews.py" script, in the function "create_dataset.py". It splits the data into train and test set and saves for each of these sets an additional file that holds the category of each word (see the line in the function that says "step 2: for each review create ...")

4. Fit vocabulary over the training set
See script "create_vocab.py", function "get_vocabulary"

5. Map OOV words to special oov token
See script "create_vocab.py", function "remove_rare"

6. Transform to word IDs
See script "create_vocab.py", function "file_to_word_ids"

7. Transform to files into numpy arrays
See script "create_vocab.py", function "split_train" and function "split_test"

Now the data is ready for training and testing!
