import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F

from const import *

class Encoder(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden = self.init_hidden()

    def forward(self, inputs, hidden):
        embedded = self.embedding(inputs).view(-1, 1, self.embedding_dim)
        output = embedded
        output, self.hidden = self.lstm(output, self.hidden)
        return output, self.hidden

    def init_hidden(self):
        return (torch.zeros(1, 1, self.hidden_dim, device=device),
                torch.zeros(1, 1, self.hidden_dim, device=device))


class Decoder(nn.Module):

    def __init__(self, out_vocab_size, embedding_dim, hidden_dim):
        super(Decoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

        # out language vocab size
        self.embedding = nn.Embedding(out_vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, out_vocab_size)
        self.hidden = self.init_hidden()
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, hidden):
        output = self.embedding(inputs).view(-1, 1, self.embedding_dim)
        output = F.relu(output)
        output, self.hidden = self.lstm(output, self.hidden)
        output = self.softmax(self.out(output[0]))
        return output, self.hidden

    def init_hidden(self):
        return (torch.zeros(1, 1, self.hidden_dim, device=device),
                torch.zeros(1, 1, self.hidden_dim, device=device))
