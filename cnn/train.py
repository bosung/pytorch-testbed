import torch
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
    try:
        return glove.vectors[glove.stoi[word]]
    except ValueError:
        return torch.FloatTensor(torch.randn(WORD_EMBED_DIM))


def train(train_data_idx):
    model = cnn.CNNTextClassifier(1, 2, WORD_EMBED_DIM, 0.5)
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    loss_function = nn.NLLLoss()

    for epoch in range(10):
        model.zero_grad()

        for i in train_data_idx:
            # input (mini_batch, in_channel, ix)
            _input = [get_glove_vector[word] for word in train_data[i][0]]
            _input = torch.cat(_input, dim=0)
            model(input)


# train_data[0] = list(sentence)
# train_data[1] = list([1, 0])
train_data = preprocessing.get_train_data()
train_data = [[sent, lable] for sent, lable in zip(train_data[0], train_data[1])]
#print(train_data[:3])
shuffle(train_data)

kf = KFold(n_split=10)
kf.get_n_splits(train_data)

for train_index, test_index in kf.split(train_data):
    print("TRAIN:", train_index, "TEST:", test_index)




#model = cnn.CNNTextClassifier(1, 2, 300, 0.2)

