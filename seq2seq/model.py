import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F

from const import *

class Encoder(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim, batch_size, weight):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size

        self.embedding = nn.Embedding.from_pretrained(weight, freeze=False)
        self.gru = nn.GRU(embedding_dim, hidden_dim)
        self.hidden = self.init_hidden(batch_size)

    def forward(self, inputs, hidden):
        embedded = self.embedding(inputs).view(MAX_LENGTH, -1, self.embedding_dim)
        output = embedded
        output, self.hidden = self.gru(output, self.hidden)
        return output, self.hidden

    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_dim, device=device)


class Decoder(nn.Module):

    def __init__(self, out_vocab_size, embedding_dim, hidden_dim, batch_size):
        super(Decoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size

        # out language vocab size
        self.embedding = nn.Embedding(out_vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, out_vocab_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, hidden):
        output = self.embedding(inputs).view(1, -1, self.embedding_dim)
        #print("decoder input: ", output.size())
        output = F.relu(output)
        output, self.hidden = self.gru(output, hidden)
        #print("decoder output 1: ", output.size())
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_dim, device=device)
