import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import torchtext.vocab as vocab
from random import shuffle
from sklearn.model_selection import KFold

import CNNTextClassifier as cnn
import preprocessing

WORD_EMBED_DIM = 300

# use glove pretrained word embedding
glove = vocab.GloVe(name='6B', dim=WORD_EMBED_DIM)


def get_glove_vector(word):
    """Return glove vector corresponding input word

    Arguments:
        word: input string

    Example:
        >>> get_glove_vector('the')
        [torch.FloatTensor of size 100]
    """
    try:
        return glove.vectors[glove.stoi[word]]
    except KeyError:
        return torch.FloatTensor(torch.randn(WORD_EMBED_DIM))


def get_sentence_matrix(words):
    """make (n, k) representaion of sentence for Conv2d
    """
    _sent_matrix = [get_glove_vector(word).view(1, 1, 1, -1) for word in words]
    _sent_matrix = torch.cat(_sent_matrix, dim=2)
    return _sent_matrix


def train(train_data_idx, drop_out, epoch_size):
    model = cnn.CNNTextClassifier(1, 2, WORD_EMBED_DIM, drop_out)
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    loss_function = nn.NLLLoss()

    for epoch in range(epoch_size):
        total_loss = 0
        for i in train_data_idx:
            model.zero_grad()
            optimizer.zero_grad()

            s_m = get_sentence_matrix(train_data[i][0])

            tag_score = model(s_m)
            target = autograd.Variable(
                    torch.LongTensor([0]) if train_data[i][1][0] == 1
                    else torch.LongTensor([1]))

            loss = loss_function(tag_score, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.data
        print("\t epoch %d\ttotal loss: %.3f" % (epoch, total_loss))

    return model


def test(model, test_data_idx):
    total_size = len(test_data_idx)
    answer_cnt = 0
    for i in test_data_idx:
        sentence, target = train_data[i][0], train_data[i][1]
        s_m = get_sentence_matrix(sentence)
        result = model(s_m).view(-1, 1)
        answer = autograd.Variable(torch.LongTensor(target))

        v, ret_idx = torch.max(result, 0)
        v, ans_idx = torch.max(answer, 0)

        if ret_idx.equal(ans_idx):
            answer_cnt += 1

    print("\t accuracy: %.3f" % (answer_cnt/total_size*100))


# train_data[0] = list(sentence)
# train_data[1] = list([1, 0])
train_data = preprocessing.get_train_data()
train_data = [[sent, lable] for sent, lable in zip(train_data[0], train_data[1])]

shuffle(train_data)

kf = KFold(n_splits=10)
kf.get_n_splits(train_data)

# hyperparameter test
drop_out, epoch_size = 0.2, 30
print("--------------------------------------------")
print("drop_out: %.1f    epoch_size: %d" % (drop_out, epoch_size))
for train_index, test_index in kf.split(train_data):
    print("TRAIN:", train_index, "TEST:", test_index)
    model = train(train_index, drop_out, epoch_size)
    test(model, test_index)

drop_out, epoch_size = 0.5, 30
print("--------------------------------------------")
print("drop_out: %.1f    epoch_size: %d" % (drop_out, epoch_size))
for train_index, test_index in kf.split(train_data):
    print("TRAIN:", train_index, "TEST:", test_index)
    model = train(train_index, drop_out, epoch_size)
    test(model, test_index)

