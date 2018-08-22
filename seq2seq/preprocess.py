from const import *
import torch
import torch.autograd as autograd

class Vocab:

    def __init__(self):
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "<SOS>", 1: "</s>", 2: "<UNK>"}
        self.n_words = 3

    def add_sentence(self, sentence):
        q, a, num = sentence.split('\t')
        for word in q.split(' '):
            self.add_word(word)
        for word in a.split(' '):
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def build(self, path):
        print("Building vocabulary from text...")

        lines = open(path, encoding='utf-8').read().strip().split('\n')
        for line in lines:
            self.add_sentence(line)

        print("[INFO] total %s words" % self.n_words)

    def load_weight(self, path="data/bobae_embedding.txt"):
        print("Loading pre-trained embeddings from %s ..." % path)
        pretrained_embedding = self._load_pretrained_embedding(path)

        include = 0
        exclude = 0
        ex_list = []
        weight = torch.tensor([]).to(device)
        for i in self.index2word.keys():
            word = self.index2word[i]
            if word in pretrained_embedding:
                vector = torch.tensor(pretrained_embedding[word])
                weight = torch.cat((weight, vector), 0)
                include += 1
            else:
                #vector = torch.randn((1, 64), dtype=torch.float).to(device)
                vector = torch.zeros((1, 64), dtype=torch.float).to(device)
                weight = torch.cat((weight, vector), 0)
                exclude += 1
                ex_list.append(word)
        print(include, exclude)
        print(ex_list)
        return torch.tensor(weight).view(-1, 64)

    def _load_pretrained_embedding(self, path):
        print("Loading embedding from %s ..." % path)

        embedding = {}
        lines = open(path, encoding='utf-8').read().strip().split('\n')
        total_num, dim = lines[0].strip().split(" ")
        for line in lines[1:]:
            toks = line.strip().split(" ")
            word = toks[0]
            vector = [float(e) for e in toks[1:]]
            embedding[word] = torch.tensor(vector, device=device).view(1, -1)
        return embedding


def read_train_data(path):
    #path = 'data/cqa_train.txt'
    lines = open(path, encoding='utf-8').read().strip().split('\n')
    pairs = [[l.split('\t')[0], l.split('\t')[1]] for l in lines]
    print("[INFO] read train data: %s ..." % pairs[0])
    return pairs


def indexesFromSentence(vocab, sentence):
    return [vocab.word2index[word] if word in vocab.word2index else UNK_token \
            for word in sentence.split(' ')][:MAX_LENGTH-1]


def tensorFromSentence(vocab, sentence):
    indexes = indexesFromSentence(vocab, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

def tensorFromSentenceBatchWithPadding(vocab, sentence_list):
    indexes = []
    for sentence in sentence_list:
        index = indexesFromSentence(vocab, sentence)
        index.append(EOS_token)

        count = MAX_LENGTH - len(index)
        for _ in range(count):
            index.append(0)

        indexes += index
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, MAX_LENGTH)

def tensorsFromPair(vocab, pair):
    input_tensor = tensorFromSentence(vocab, pair[0])
    target_tensor = tensorFromSentence(vocab, pair[1])
    return (input_tensor, target_tensor)

