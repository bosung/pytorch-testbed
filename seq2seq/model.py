import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F

from const import *


class Encoder(nn.Module):

    def __init__(self, vocab_size, embed_size, hidden_size, batch_size, weight=None):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.batch_size = batch_size

        if weight is None:
            self.embedding = nn.Embedding(vocab_size, embed_size)
        else:
            self.embedding = nn.Embedding.from_pretrained(weight, freeze=False)

        self.gru = nn.GRU(embed_size, hidden_size)

    def forward(self, inputs, hidden):
        embedded = self.embedding(inputs)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size, device=device)


class Decoder(nn.Module):

    def __init__(self, out_vocab_size, embed_size, hidden_size, batch_size, weight=None):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.batch_size = batch_size

        if weight is None:
            self.embedding = nn.Embedding(out_vocab_size, embed_size)
        else:
            self.embedding = nn.Embedding.from_pretrained(weight, freeze=False)

        self.gru = nn.GRU(embed_size, hidden_size)
        self.out = nn.Linear(hidden_size, out_vocab_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, hidden):
        output = self.embedding(inputs).view(1, -1, self.embed_size)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        # output.size() = ([1, batch_size, hidden_size])
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size, device=device)


class AttentionDecoder(nn.Module):

    def __init__(self, out_vocab_size, embed_size, hidden_size, batch_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttentionDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.batch_size = batch_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        # out language vocab size
        self.embedding = nn.Embedding(out_vocab_size, self.hidden_size)
        self.attention = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attention_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, out_vocab_size)

    def forward(self, inputs, hidden, encoder_outputs):
        embedded = self.embedding(inputs).view(1, -1, self.hidden_size)
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

