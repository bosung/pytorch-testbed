import evaluate as ev
import preprocess as prep

import torch
import torch.nn as nn

from preprocess import Vocab
from utils import *

TRAIN_PATH = 'data/cqa_train_komoran.txt'
#EMBED_PATH = 'data/hd_embedding.txt'
#EMBED_PATH = 'data/komoran_news_embedding.txt'
EMBED_PATH = 'data/komoran_hd_2times.vec'

softmax = nn.Softmax(dim=0)

def get_sent_embed(sentence, pre_trained_embedding):
    ''' get sentence embedding '''

    x = pre_trained_embedding[sentence[0]].view(1, -1)
    for i in sentence[1:]:
        x = torch.cat((x, pre_trained_embedding[i].view(1, -1)), 0)

    # sent_matrix.size() = torch.Size([n, d])
    attn_matrix = torch.matmul(x, x.transpose(0, 1))
    result = softmax(attn_matrix)
    self_attn_matrix = torch.matmul(result, x)

    # represent sentece by averaging matrix
    applied_sent = torch.mean(self_attn_matrix, 0)
    return applied_sent


def get_embed(vocab, data, pre_trained_embedding):
    embed = {}
    sentences = {}
    for k in data:
        sentences[k] = prep.tensorFromSentence(vocab, data[k])

    for k in sentences:
        sentence = sentences[k]
        # make sentence matrix
        embed[k] = get_sent_embed(sentence, pre_trained_embedding)

    return embed


def evaluate():
    vocab = Vocab()
    vocab.build(TRAIN_PATH)
    # torch.tensor([2764, 64])
    pre_trained_embedding = vocab.load_weight(EMBED_PATH)

    train_data, test_data, test_answer = ev.prepare_evaluate()

    train_embed = get_embed(vocab, train_data, pre_trained_embedding)

    # evaluation
    print("[INFO] start evaluating!")
    total = len(test_data)
    answer5 = 0
    answer1 = 0

    for tk in test_data:
        print("Q.%s %s" % (tk, pretty_printer2(test_data[tk])))
        test_in = prep.tensorFromSentence(vocab, test_data[tk])
        embedded = get_sent_embed(test_in, pre_trained_embedding)

        temp = {}
        for candi in train_embed.keys():
            t = train_embed[candi]
            e = embedded
            temp[candi] = cosine_similarity(t, e)

        top_n = get_top_n(temp, 5)
        for e in top_n.keys():
            print("%.8f %4s %s" % (top_n[e], e, pretty_printer2(train_data[e])))
            if ev.isAnswer(e, test_answer[tk]):
                answer5 += 1
                break
        top1 = list(top_n.keys())[0]
        if ev.isAnswer(top1, test_answer[tk]):
            answer1 += 1
        print("------------------------------------------")

    accuracy_at_5 = answer5/total*100
    accuracy_at_1 = answer1/total*100

    print("total: %d, accuracy@5: %.4f, accuracy@1: %.4f" % (total, accuracy_at_5, accuracy_at_1))


def print_vectors():
    vocab = Vocab()
    vocab.build(TRAIN_PATH)

    # torch.tensor([2764, 64])
    pre_trained_embedding = vocab.load_weight(EMBED_PATH)

    train_data, test_data, test_answer = ev.prepare_evaluate()

    obj = train_data

    for d in obj.keys():
        t_in = prep.tensorFromSentence(vocab, obj[d])
        embedded = get_sent_embed(t_in, pre_trained_embedding)
        print("%s\t%s" % (d, ' '.join([str(e) for e in embedded.squeeze().data.tolist()])))


if __name__=="__main__":
    #evaluate()
    print_vectors()

