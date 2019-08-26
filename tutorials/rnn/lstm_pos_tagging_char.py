import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
import torch.optim as optim


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    tensor = torch.LongTensor(idxs)
    return autograd.Variable(tensor)

training_data = [
    ("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
    ("Everybody read that book".split(), ["NN", "V", "DET", "NN"])
]
word_to_ix = {}
for sent, tags in training_data:
    for word in sent:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
print(word_to_ix)
tag_to_ix = {"DET": 0, "NN": 1, "V": 2}

WORD_EMBED_DIM = 5
CHAR_EMBED_DIM = 3
HIDDEN_DIM = 6

class LSTMTaggerCharLevel(nn.Module):

    def __init__(self, word_embed_dim, char_embed_dim, hidden_dim,
            vocab_size, target_size):
        super(LSTMTaggerCharLevel, self).__init__()
        self.hidden_dim = hidden_dim
        self.char_embed_dim = char_embed_dim

        self.word_embeddings = nn.Embedding(vocab_size, word_embed_dim)
        # alphabet with capital, lower case + exception
        self.char_embeddings = nn.Embedding(53, char_embed_dim)

        self.lstm = nn.LSTM(word_embed_dim+char_embed_dim, hidden_dim)
        self.lstm_char = nn.LSTM(char_embed_dim, char_embed_dim)

        self.hidden_for_char = self.init_hidden(char_embed_dim)
        self.hidden = self.init_hidden(hidden_dim)

        self.hidden2tag = nn.Linear(hidden_dim, target_size)

    def forward(self, sentence):
        # sentence: ["The", "dog", ..., "apple"]
        #
        self.hidden = self.init_hidden(self.hidden_dim)

        concated_inputs = []
        for word in sentence:
            self.hidden_for_char = self.init_hidden(self.char_embed_dim)
            pre_char_embeds = self.char_embeddings(self.get_char_idx(word))
            # print(word, pre_char_embeds)
            # print(pre_char_embeds.view(len(word), 1, -1))

            char_lstm_out, self.hidden_for_char = self.lstm_char(
                    pre_char_embeds.view(len(word), 1, -1), self.hidden_for_char)

            # print(word, char_lstm_out, self.hidden_for_char)
            # print("========================")

            # Cw be the final hidden state of char-level LSTM
            char_embed = self.hidden_for_char[0]

            # concat Xw and Cw
            # step 1. get Xw
            word_embed = self.word_embeddings(
                    autograd.Variable(torch.LongTensor([word_to_ix[word]])))
            # print(char_embed, word_embed.view(1, 1, -1))

            concated_input = torch.cat((word_embed.view(1, 1, -1), char_embed), 2)
            # print(concated_input)
            concated_inputs.append(concated_input)

        concated_sentence = torch.cat(concated_inputs, 0)

        lstm_out, self.hidden = self.lstm(concated_sentence, self.hidden)
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_score = F.log_softmax(tag_space, dim=1)
        return tag_score

    def get_char_idx(self, word):
        char_idx = []
        for c in word:
            asc_code = ord(c)
            if ord('A') <= asc_code <= ord('Z'):
                char_idx.append(asc_code-ord('A'))
            elif ord('a') <= asc_code <= ord('z'):
                char_idx.append(asc_code-71)
            else:
                char_idx.append(52)
        tensor = torch.LongTensor(char_idx)
        return autograd.Variable(tensor)

    def init_hidden(self, hid_dim):
        return (autograd.Variable(torch.zeros(1, 1, hid_dim)),
                autograd.Variable(torch.zeros(1, 1, hid_dim)))


model = LSTMTaggerCharLevel(WORD_EMBED_DIM, CHAR_EMBED_DIM, HIDDEN_DIM,
        len(word_to_ix), len(tag_to_ix))
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

#model(training_data[0][0])

# get_char_idx function test
#print(model.get_char_idx("Aa"))
#print(model.get_char_idx("the"))
#print(model.get_char_idx("The"))
#print(model.get_char_idx("That"))
for epoch in range(300):
    total_loss = 0
    for sentence, tags in training_data:
        model.zero_grad()
        # ?
        # model.hidden = model.init_hidden()
        tag_scores = model(sentence)

        targets = prepare_sequence(tags, tag_to_ix)
        loss = loss_function(tag_scores, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.data
    print(total_loss)

