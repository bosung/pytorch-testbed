import numpy as np
import re

def clean_str(string, remove_dot=True):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    if remove_dot:
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    else:
        string = re.sub(r"[^A-Za-z0-9().,!?\'\`]", " ", string)

    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    # string = re.sub(r"\(", " \( ", string)
    # string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data(positive_data_file, negative_data_file):
    positive_examples = list(open(positive_data_file, "r").readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(negative_data_file, "r").readlines())
    negative_examples = [s.strip() for s in negative_examples]

    # split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    x_text = [sent.split(' ') for sent in x_text]

    pos_labels = [[0, 1] for _ in positive_examples]
    neg_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([pos_labels, neg_labels], 0)

    return [x_text, y]


def get_train_data():
    data_dir = '/home/nlp908/bosung/data/rt-polarity'
    pos_file = '/rt-polarity.pos.txt'
    neg_file = '/rt-polarity.neg.txt'

    return load_data(data_dir+pos_file, data_dir+neg_file)

