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

        if weight.nelement() == 0:
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
        else:
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

    def __init__(self, out_vocab_size, embedding_dim, hidden_dim, batch_size, weight):
        super(Decoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size

        if weight.nelement() == 0:
            self.embedding = nn.Embedding(out_vocab_size, embedding_dim)
        else:
            self.embedding = nn.Embedding.from_pretrained(weight, freeze=False)

        self.gru = nn.GRU(embedding_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, out_vocab_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, hidden):
        output = self.embedding(inputs).view(1, -1, self.embedding_dim)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_dim, device=device)


class AttentionDecoder(nn.Module):

    def __init__(self, out_vocab_size, embedding_dim, hidden_dim, batch_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttentionDecoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        # out language vocab size
        self.embedding = nn.Embedding(out_vocab_size, self.hidden_dim)
        self.attention = nn.Linear(self.hidden_dim * 2, self.max_length)
        self.attention_combine = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, out_vocab_size)

    def forward(self, inputs, hidden, encoder_outputs):
        embedded = self.embedding(inputs).view(1, -1, self.hidden_dim)
        embedded - self.dropout(embedded)

        attention_weights = F.softmax(
                self.attention(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        a_w = attention_weights.unsqueeze(1)

        e_o = encoder_outputs.transpose(0, 1)
        attention_applied = torch.bmm(a_w, e_o)

        output = torch.cat((embedded[0], attention_applied.transpose(0, 1)[0]), 1)
        output = self.attention_combine(output).unsqueeze(0)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attention_weights

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

