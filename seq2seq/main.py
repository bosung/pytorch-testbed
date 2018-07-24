from model import Encoder, Decoder, AttentionDecoder
from preprocess import Vocab
from utils import asMinutes, timeSince, cosine_similarity, get_top_n
import preprocess as prep
import evaluate as ev

import argparse
import time
import torch
import torch.nn as nn
import torch.utils.data as data
from torch import optim
import random

from const import *

global batch_size

def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):

    encoder_hidden = encoder.init_hidden(max_length)

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    #encoder_outputs = torch.zeros(max_length, encoder.hidden_dim, device=device)

    loss = 0

    #input_tensor = input_tensor.transpose(0, 1)
    target_tensor = target_tensor.transpose(0, 1)

    encoder_outputs, encoder_hidden = encoder(input_tensor, encoder_hidden)

    decoder_input = torch.tensor([[SOS_token] * batch_size], device=device).view(1, batch_size)
    decoder_hidden = encoder_hidden

    for di in range(max_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)

        loss += criterion(decoder_output, target_tensor[di])
        decoder_input = target_tensor[di]  # Teacher forcing

    loss.backward(retain_graph=True)

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / max_length


def trainIters(epoch, encoder, decoder, n_iters, pairs, vocab, train_loader,
        print_every=1000, plot_every=100, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    #encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    #decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

    criterion = nn.NLLLoss()

    num_iters = 0
    for _iter, (batch_input, batch_target) in enumerate(train_loader):
        input_tensor = prep.tensorFromSentenceBatchWithPadding(vocab, batch_input)
        target_tensor = prep.tensorFromSentenceBatchWithPadding(vocab, batch_target)

        loss = train(input_tensor, target_tensor, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss
        num_iters += batch_size

    if epoch % print_every == 0:
        print_loss_avg = print_loss_total / num_iters
        print_loss_total = 0
        print('%s (%d %d%%) %.4f' % (timeSince(start, epoch / n_iters),
                                epoch, epoch / n_iters * 100, print_loss_avg))



def evaluate(encoder, decoder, sentence, vocab, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = prep.tensorFromSentenceBatchWithPadding(vocab, sentence)

        encoder_hidden = encoder.init_hidden(max_length)

        #encoder_outputs = torch.zeros(max_length, encoder.hidden_dim, device=device)

        input_tensor = input_tensor.transpose(0, 1)
        encoder_output, encoder_hidden = encoder(input_tensor, encoder_hidden)

        decoder_input = torch.tensor([[SOS_token] * batch_size], device=device).view(1, batch_size)  # SOS
        decoder_hidden = encoder_hidden

        decoded_words_batch = []
        for _ in range(batch_size):
            decoded_words_batch.append([])

        #print(decoder_input)
        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_output)
            topv, topi = decoder_output.data.topk(1)
            decoder_input = topi.squeeze().view(1, batch_size)

            #print(decoder_input)
            #print(decoder_output.size())
            for i, out in enumerate(decoder_output):
                top = out.data.topk(1)[1]
                #print(top.item())
                if top.item() == EOS_token:
                    decoded_words_batch[i].append('<EOS>')
                else:
                    decoded_words_batch[i].append(vocab.index2word[top.item()])

        #return decoded_words, decoder_attentions[:di + 1]
        #print(decoded_words_batch)
        return decoded_words_batch

def pretty_printer(data):
    return [x.split("/")[0] for x in data]

def pretty_printer2(data):
    return ' '.join([x.split("/")[0] for x in data.split(" ")][:MAX_LENGTH])

def evaluateRandomly(encoder, decoder, pairs, vocab, batch_size, n=10):
    test = [p[0] for p in pairs][:batch_size]
    answer = [p[1] for p in pairs][:batch_size]
    result_batch = evaluate(encoder, decoder, test, vocab)

    for i, out in enumerate(result_batch):
    #for pair in pairs:
        #pair = random.choice(pairs)
        print('>', pretty_printer2(test[i]))
        print('=', pretty_printer2(answer[i]))
        #output_words = evaluate(encoder, decoder, pair[0], vocab)
        output_sentence = ' '.join(pretty_printer(out))
        print('<', output_sentence)
        print('')

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder', help='load exisited model')
    parser.add_argument('--decoder', help='load exisited model')
    parser.add_argument('--batch_size', default=20)
    args = parser.parse_args()

    train_file = 'data/cqa_train.txt'

    vocab = Vocab()
    vocab.build(train_file)
    if args.encoder:
        weight = empty_weight
    else:
        # load pre-trained embedding
        weight = vocab.load_weight()

    global batch_size
    batch_size = args.batch_size

    train_data = prep.read_train_data(train_file)
    train_loader = data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

    encoder = Encoder(vocab.n_words, 64, 32, batch_size, weight).to(device)
    decoder = AttentionDecoder(vocab.n_words, 64, 32, batch_size).to(device)

    if args.encoder:
        encoder.load_state_dict(torch.load(args.encoder))
    if args.decoder:
        decoder.load_state_dict(torch.load(args.decoder))

    #ev.evaluate(encoder, vocab, batch_size)

    # max accuracy
    max_a_at_5 = 0
    max_a_at_1 = 0

    total_epoch = 400

    for epoch in range(1, total_epoch+1):
        random.shuffle(train_data)
        trainIters(epoch, encoder, decoder, total_epoch, train_data, vocab, train_loader, print_every=5, learning_rate=0.01)

        if epoch % 20 == 0:
            a_at_5, a_at_1 = ev.evaluate(encoder, vocab, batch_size)
            #evaluateRandomly(encoder, decoder, train_data, vocab, batch_size)

            if a_at_1 > max_a_at_1:
                max_a_at_1 = a_at_1
                print("New record! accuracy@1: %.4f" % a_at_1)
                torch.save(encoder.state_dict(), 'encoder.model')
                torch.save(decoder.state_dict(), 'decoder.model')
                evaluateRandomly(encoder, decoder, train_data, vocab, batch_size)


