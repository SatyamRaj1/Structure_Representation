# _*_ coding: utf-8 _*_

import os
import sys
import torch
from torch.nn import functional as F
import numpy as np
from torchtext import data
from torchtext import datasets
#from torchtext.vocab import Vectors, GloVe

def load_dataset(file_path_train='toxicbias_train_aug.csv', file_path_test='toxicbias_test_aug.csv', batch_size=32):
    
    """
    tokenizer : Breaks sentences into a list of words. If sequential=False, no tokenization is applied
    Field : A class that stores information about the way of preprocessing
    fix_length : An important property of TorchText is that we can let the input to be variable length, and TorchText will
                 dynamically pad each sequence to the longest sequence in that "batch". But here we are using fi_length which
                 will pad each sequence to have a fix length of 200.
                 
    build_vocab : It will first make a vocabulary or dictionary mapping all the unique words present in the train_data to an
                  idx and then after it will use GloVe word embedding to map the index to the corresponding word embedding.
                  
    vocab.vectors : This returns a torch tensor of shape (vocab_size x embedding_dim) containing the pre-trained word embeddings.
    BucketIterator : Defines an iterator that batches examples of similar lengths together to minimize the amount of padding needed.
    
    """
    
    
    # Load data from CSV
    df_train = pd.read_csv(file_path_train)
    df_test = pd.read_csv(file_path_test)

    # Select only the necessary columns and process the 'bias' column
    df_train = df_train[['comment_text', 'bias']]
    df_test = df_test[['comment_text', 'bias']]
    df_train['bias'] = df_train['bias'].apply(lambda x: 1 if x == 'bias' else 0)
    df_test['bias'] = df_test['bias'].apply(lambda x: 1 if x == 'bias' else 0)

    # Save processed CSV (for torchtext)
    train_csv_path = "processed_train.csv"
    test_csv_path = "processed_test.csv"
    df_train.to_csv(train_csv_path, index=False)
    df_test.to_csv(test_csv_path, index=False)

    # Tokenizer from transformers
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    # Define preprocessing pipeline
    def tokenize_and_cut(sentence):
        tokens = tokenizer.tokenize(sentence)
        tokens = tokens[:198]  # Reduce length by 2 to account for special tokens
        return tokens

    TEXT = Field(sequential=True,
                 use_vocab=False,
                 tokenize=tokenize_and_cut,
                 preprocessing=tokenizer.convert_tokens_to_ids,
                 init_token=tokenizer.cls_token_id,
                 eos_token=tokenizer.sep_token_id,
                 pad_token=tokenizer.pad_token_id,
                 unk_token=tokenizer.unk_token_id,
                 include_lengths=True,
                 batch_first=True)

    LABEL = LabelField(dtype=torch.float)

    # Define fields mapping from CSV to fields
    fields = {'comment_text': ('text', TEXT), 'bias': ('label', LABEL)}

    # Create dataset using the paths to the CSV files
    train_data, test_data = TabularDataset.splits(
                                path='',
                                train=train_csv_path,
                                test=test_csv_path,
                                format='csv',
                                fields=fields)

    # Build vocabulary for labels
    LABEL.build_vocab(train_data)

    # Print the outputs
    print("Length of Text Vocabulary: " + str(tokenizer.vocab_size))
    print("Label Length: " + str(len(LABEL.vocab)))

    # Create iterators
    train_iter, test_iter = BucketIterator.splits(
        (train_data, test_data),
        batch_size=batch_size,
        sort_within_batch=True,
        sort_key=lambda x: len(x.text))

    return TEXT, tokenizer.vocab_size, None, train_iter, None, test_iter

