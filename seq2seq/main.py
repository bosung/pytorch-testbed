from model import Encoder, Decoder
from preprocess import Vocab
from utils import asMinutes, timeSince, cosine_similarity, get_top_n
import preprocess as prep
import evaluate as ev

import time
import torch
import torch.nn as nn
from torch import optim
import random

from const import *


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    encoder_hidden = encoder.init_hidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_dim, device=device)

    loss = 0

    for ei in range(min(input_length, max_length)):
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)
    decoder_hidden = encoder_hidden

    #print(input_length)
    #print(encoder_hidden)
    #print(encoder_hidden.size())

    for di in range(target_length):
        decoder_output, decoder_hidden = decoder(
            decoder_input, decoder_hidden)
        loss += criterion(decoder_output, target_tensor[di])
        decoder_input = target_tensor[di]  # Teacher forcing

    loss.backward(retain_graph=True)

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


def trainIters(encoder, decoder, n_iters, pairs, vocab,
        print_every=1000, plot_every=100, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    training_pairs = [prep.tensorsFromPair(vocab, pairs[i]) for i in range(len(pairs))]
    criterion = nn.NLLLoss()

    # epoch = n_iters
    for iter in range(1, n_iters + 1):
        random.shuffle(training_pairs)
        for pair in training_pairs:
            #training_pair = training_pairs[iter - 1]
            input_tensor = pair[0]
            target_tensor = pair[1]

            #print(input_tensor.view(1,-1))
            #print(target_tensor.view(1, -1))

            loss = train(input_tensor, target_tensor, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion)
            print_loss_total += loss
            plot_loss_total += loss


        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

        #if iter % 5 == 0:
        #    ev.evaluate(encoder, vocab)


def evaluate(encoder, decoder, sentence, vocab, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = prep.tensorFromSentence(vocab, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.init_hidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_dim, device=device)

        for ei in range(min(max_length, input_length)):
            encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        #decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)
            #decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(vocab.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        #return decoded_words, decoder_attentions[:di + 1]
        return decoded_words

def pretty_printer(data):
    return [x.split("/")[0] for x in data]

def pretty_printer2(data):
    return ' '.join([x.split("/")[0] for x in data.split(" ")])

def evaluateRandomly(encoder, decoder, pairs, vocab, n=5):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pretty_printer2(pair[0]))
        print('=', pretty_printer2(pair[1]))
        output_words = evaluate(encoder, decoder, pair[0], vocab)
        output_sentence = ' '.join(pretty_printer(output_words))
        print('<', output_sentence)
        print('')


if __name__=="__main__":
    train_file = 'data/cqa_train_temp.txt'

    vocab = Vocab()
    vocab.build(train_file)

    train_data = prep.read_train_data(train_file)

    encoder = Encoder(vocab.n_words, 16, 32).to(device)
    decoder = Decoder(vocab.n_words, 16, 32).to(device)

    trainIters(encoder, decoder, 100, train_data, vocab, print_every=5, learning_rate=0.01)

    evaluateRandomly(encoder, decoder, train_data, vocab)

    #evaluate(encoder, vocab)

