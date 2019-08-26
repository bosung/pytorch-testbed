import math
import matplotlib.pyplot as plt
import numpy as np
import preprocess as prep
from preprocess import Vocab
import seaborn as sns
import torch
import torch.nn.functional as F

EMBED_PATH = 'data/komoran_hd_2times.vec'
train_file = 'data/train_data_nv.txt'

np.random.seed(0)
sns.set()


def scaled_dot_product_attn(Q, K, V, scale=False, softmax=False):
    attn_matrix = torch.matmul(Q, K.transpose(0, 1))

    if scale:
        d_k = attn_matrix.size(0)
        attn_matrix = torch.mul(attn_matrix, math.sqrt(d_k))

    if softmax:
        attn_matrix = F.softmax(attn_matrix, dim=0)

    return torch.matmul(attn_matrix, V)


def get_word_embed_matrix(vocab, sentence, _pre_trained_embedding):
    """ return sentence matrix. each row is word embedding """
    test_in = prep.tensorFromSentence(vocab, sentence)

    x = _pre_trained_embedding[test_in[0]].view(1, -1)
    for i in test_in[1:]:
        x = torch.cat((x, _pre_trained_embedding[i].view(1, -1)), 0)

    return x


def get_sentence_embed(vocab, sentence, pre_trained_embedding):
    """ represent sentence by averaing word embeddings """
    we_matrix = get_word_embed_matrix(sentence, vocab, pre_trained_embedding)
    return torch.mean(we_matrix, 0)


def get_sentence_embed_sa(vocab, sentence, pre_trained_embedding):
    we_matrix = get_word_embed_matrix(vocab, sentence, pre_trained_embedding)
    applied_sent = scaled_dot_product_attn(we_matrix, we_matrix, we_matrix)
    return applied_sent


if __name__=="__main__":
    vocab = Vocab()
    vocab.build(train_file)
    pre_trained_embedding = vocab.load_weight(EMBED_PATH)

    sentence = '에어컨/NNG 작동/NNG 시/NNB 냉방/NNG 성능/NNG 떨어지/VV 그렇/VA 모르/VV 어떻/VA 하/VV 하/VX'
    #sentence = '에어컨/NNG 시원/XR 나오/VV 않/VX 그렇/VA 자동차/NNG 고장/NNG 아니/VCN 하/VV 연락/NNG 드리/VV'
    #data = get_word_embed_matrix(vocab, sentence, pre_trained_embedding)
    data = get_sentence_embed_sa(vocab, sentence, pre_trained_embedding)
    print(data.size())
    sns.heatmap(data)
    plt.show()

