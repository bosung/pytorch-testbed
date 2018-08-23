import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F

from const import *


class Encoder(nn.Module):

    def __init__(self, vocab_size, embed_size, hidden_size, batch_size, embedding_weight=None):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.batch_size = batch_size

        if embedding_weight is None:
            self.embedding = nn.Embedding(vocab_size, embed_size)
        else:
            self.embedding = nn.Embedding.from_pretrained(embedding_weight, freeze=False)

        self.gru = nn.GRU(embed_size, hidden_size, batch_first=True)

    def forward(self, inputs, hidden):
        embedded = self.embedding(inputs)
        output, hidden = self.gru(embedded, hidden)
        return output, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size, device=device)


class Decoder(nn.Module):

    def __init__(self, out_vocab_size, embed_size, hidden_size, batch_size, embedding_weight=None):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.batch_size = batch_size

        if embedding_weight is None:
            self.embedding = nn.Embedding(vocab_size, embed_size)
        else:
            self.embedding = nn.Embedding.from_pretrained(embedding_weight, freeze=False)

        self.gru = nn.GRU(embed_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, out_vocab_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, hidden):
        output = self.embedding(inputs).view(-1, 1, self.embed_size)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = output.transpose(1, 2)
        output = self.softmax(self.out(output.view(-1, self.hidden_size)))
        return output, hidden


class AttentionDecoder(nn.Module):
    """
    Apply attention based on https://arxiv.org/pdf/1508.04025.pdf
    """

    def __init__(self, out_vocab_size, embed_size, hidden_size, batch_size, embedding_weight=None, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttentionDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.batch_size = batch_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        if embedding_weight is None:
            self.embedding = nn.Embedding(out_vocab_size, embed_size)
        else:
            self.embedding = nn.Embedding.from_pretrained(embedding_weight, freeze=False)

        self.attention = nn.Linear(self.hidden_size, self.hidden_size)
        self.attention_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(embed_size, hidden_size)
        self.out = nn.Linear(hidden_size, out_vocab_size)

    def forward(self, inputs, hidden, encoder_outputs):
        embedded = self.embedding(inputs).view(1, -1, self.hidden_size)
        embedded - self.dropout(embedded)

        # step 1. GRU
        gru_out, hidden = self.gru(embedded, hidden)

        print(encoder_outputs.size()) # (15, 40, 128)
        print(hidden.size()) # (40, 128)

        # step 2. general score
        for e in range(encoder_outputs.size(0)):
            attn_prod = torch.bmm(hidden, self.attention(encoder_outputs))
            # attn_prod = (40, 15)
        attention_weights = F.softmax(attn_prod, dim=1)

        a_w = attention_weights.unsqueeze(1)

        e_o = encoder_outputs.transpose(0, 1)
        attention_applied = torch.bmm(a_w, e_o)

        output = torch.cat((embedded[0], attention_applied.transpose(0, 1)[0]), 1)
        output = self.attention_combine(output).unsqueeze(0)
        output = F.relu(output)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attention_weights

