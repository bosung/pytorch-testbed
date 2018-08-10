import preprocess as prep
from const import *
#from utils import cosine_similarity
from utils import *

import torch
import torch.nn as nn

from nltk.translate.bleu_score import sentence_bleu

softmax = nn.Softmax(dim=0)

def prepare_evaluate():
    train_data = {}
    test_data = {}
    test_answer = {}

    lines = open('data/train_list_komoran.txt', 'r').read().strip().split('\n')
    for l in lines:
        q, num = l.split('\t')
        train_data[num] = q

    lines = open('data/test_list_komoran.txt', 'r').read().strip().split('\n')
    for l in lines:
        q, num, answer = l.split('\t')
        test_data[num] = q
        test_answer[num] = answer

    return train_data, test_data, test_answer


def get_embed(encoder, sentence, vocab, batch_size, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = prep.tensorFromSentenceBatchWithPadding(vocab, [sentence])

        # for batch, need expansion for input tensor
        temp = input_tensor
        for _ in range(batch_size-1):
            temp = torch.cat((temp, input_tensor), 0)
        input_tensor = temp

        input_tensor = input_tensor.transpose(0, 1)

        encoder_hidden = encoder.init_hidden(batch_size)

        encoder_output, encoder_hidden = encoder(input_tensor, encoder_hidden)

        # consider last encoder_hidden as sentence embedding
        return encoder_hidden[0][0].view(1, 1, -1)


def get_embed_concat(encoder, decoder, sentence, vocab, batch_size, max_length=MAX_LENGTH):
    """ sentence embedding test
        v1. concat two hidden vector of encoder and decoder
    """
    with torch.no_grad():
        input_tensor = prep.tensorFromSentenceBatchWithPadding(vocab, [sentence])

        # because of batch, need expansion for input tensor
        temp = input_tensor
        for _ in range(batch_size-1):
            temp = torch.cat((temp, input_tensor), 0)
        input_tensor = temp

        encoder_hidden = encoder.init_hidden(max_length)

        encoder_output, encoder_hidden = encoder(input_tensor, encoder_hidden)

        decoder_input = torch.tensor([[SOS_token] * batch_size], device=device).view(1, batch_size)  # SOS
        decoder_hidden = encoder_hidden

        decoded_words_batch = []
        for _ in range(batch_size):
            decoded_words_batch.append([])

        #print(decoder_input)
        for di in range(max_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            #print(decoder_hidden.squeeze()[0][:7])
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

        # concat two hidden vector of encoder, decoder
        C_Q = encoder_hidden[0][0].view(1, -1)
        C_A = decoder_hidden[0][0].view(1, -1)
        return torch.cat((C_Q, C_A), 0)


def get_embed_q_pivot(encoder, decoder, sentence, vocab, batch_size, max_length=MAX_LENGTH):
    """ sentence embedding test
        v2. answer attentioned vector in light of question vector
    """
    with torch.no_grad():
        input_tensor = prep.tensorFromSentenceBatchWithPadding(vocab, [sentence])

        # because of batch, need expansion for input tensor
        temp = input_tensor
        for _ in range(batch_size-1):
            temp = torch.cat((temp, input_tensor), 0)
        input_tensor = temp

        encoder_hidden = encoder.init_hidden(max_length)

        encoder_output, encoder_hidden = encoder(input_tensor, encoder_hidden)

        decoder_input = torch.tensor([[SOS_token] * batch_size], device=device).view(1, batch_size)  # SOS
        decoder_hidden = encoder_hidden

        for di in range(max_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            topv, topi = decoder_output.data.topk(1)
            decoder_input = topi.squeeze().view(1, batch_size)

        C_Q = encoder_hidden[0][0].view(1, -1)
        C_A = decoder_hidden[0][0].view(1, -1)

        L = torch.matmul(C_A.transpose(0, 1), C_Q)
        A_Q = softmax(L)
        C_QA = torch.matmul(C_A, A_Q)

        return C_QA


def get_embed_ans_pivot(encoder, decoder, sentence, vocab, batch_size, max_length=MAX_LENGTH):
    """ sentence embedding test
        v2. answer attentioned vector in light of question vector
    """
    with torch.no_grad():
        input_tensor = prep.tensorFromSentenceBatchWithPadding(vocab, [sentence])

        # because of batch, need expansion for input tensor
        temp = input_tensor
        for _ in range(batch_size-1):
            temp = torch.cat((temp, input_tensor), 0)
        input_tensor = temp

        encoder_hidden = encoder.init_hidden(max_length)

        encoder_output, encoder_hidden = encoder(input_tensor, encoder_hidden)

        decoder_input = torch.tensor([[SOS_token] * batch_size], device=device).view(1, batch_size)  # SOS
        decoder_hidden = encoder_hidden

        for di in range(max_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            topv, topi = decoder_output.data.topk(1)
            decoder_input = topi.squeeze().view(1, batch_size)

        C_Q = encoder_hidden[0][0].view(1, -1)
        C_A = decoder_hidden[0][0].view(1, -1)

        L = torch.matmul(C_Q.transpose(0, 1), C_A)
        A_A = softmax(L)
        C_AQ = torch.matmul(C_Q, A_A)

        return C_AQ


def evaluate_similarity(encoder, vocab, batch_size, decoder=None):
    train_list, test_list, test_answer = prepare_evaluate()

    # embed candidates
    train_embed = {}
    print("[INFO] encoding train %d data ..." % len(train_list))
    for d in train_list.keys():
        embedded = get_embed(encoder, train_list[d], vocab, batch_size)
        #embedded = get_embed_concat(encoder, decoder, train_list[d], vocab, batch_size)
        train_embed[d] = embedded
    print("[INFO] done")

    print("[INFO] start evaluating!")
    total = len(test_list)
    answer5 = 0
    answer1 = 0
    for i, tk in enumerate(test_list.keys()):
        embedded = get_embed(encoder, test_list[tk], vocab, batch_size)
        #embedded = get_embed_concat(encoder, decoder, train_list[d], vocab, batch_size)
        temp = {}
        for candi in train_embed.keys():
            t = train_embed[candi].view(encoder.hidden_size)
            e = embedded.view(encoder.hidden_size)
            temp[candi] = cosine_similarity(t, e)

        top_n = get_top_n(temp, 5)
        for e in top_n.keys():
            if isAnswer(e, test_answer[tk]):
                answer5 += 1
                break
        top1 = list(top_n.keys())[0]
        if isAnswer(top1, test_answer[tk]):
            answer1 += 1

    accuracy_at_5 = answer5/total*100
    accuracy_at_1 = answer1/total*100

    print("total: %d, accuracy@5: %.4f, accuracy@1: %.4f" % (total, accuracy_at_5, accuracy_at_1))

    return accuracy_at_5, accuracy_at_1


def evaluate(encoder, decoder, sentence, vocab, batch_size, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = prep.tensorFromSentenceBatchWithPadding(vocab, sentence)

        encoder_hidden = encoder.init_hidden(batch_size)
        #encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        input_tensor = input_tensor.transpose(0, 1)
        encoder_output, encoder_hidden = encoder(input_tensor, encoder_hidden)

        decoder_input = torch.tensor([[SOS_token] * batch_size], device=device).view(1, batch_size)  # SOS
        decoder_hidden = encoder_hidden

        decoded_words_batch = []
        for _ in range(batch_size):
            decoded_words_batch.append([])

        #print(decoder_input)
        for di in range(max_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
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


def evaluateRandomly(encoder, decoder, pairs, vocab, batch_size, n=10):
    test = [p[0] for p in pairs][:batch_size]
    answer = [p[1] for p in pairs][:batch_size]
    result_batch = evaluate(encoder, decoder, test, vocab, batch_size)

    total_bleu_score = 0
    for i, out in enumerate(result_batch):
    #for pair in pairs:
        #pair = random.choice(pairs)
        bleu_score = sentence_bleu([out], answer[i].split(" ")[:MAX_LENGTH])
        total_bleu_score += bleu_score

        print("sentence bleu score: %.2f" % bleu_score)
        print('>', pretty_printer2(test[i]))
        print('=', pretty_printer2(answer[i]))
        #output_words = evaluate(encoder, decoder, pair[0], vocab)
        output_sentence = ' '.join(pretty_printer(out))
        print('<', output_sentence)
        print('')
    avg_bleu = total_bleu_score/len(result_batch)
    print("Average BLEU score: %.2f" % avg_bleu)
    return avg_bleu


def evaluate_with_print(encoder, vocab, batch_size):
    train_list, test_list, test_answer = prepare_evaluate()

    # embed candidates
    train_embed = {}
    print("[INFO] encoding train %d data ..." % len(train_list))
    for d in train_list.keys():
        embedded = get_embed(encoder, train_list[d], vocab, batch_size)
        train_embed[d] = embedded
    print("[INFO] done")

    print("[INFO] start evaluating!")
    total = len(test_list)
    answer5 = 0
    answer1 = 0
    for i, tk in enumerate(test_list.keys()):
        print("Q%d. %s" % (i+1, pretty_printer2(test_list[tk])))
        embedded = get_embed(encoder, test_list[tk], vocab, batch_size)
        temp = {}
        for candi in train_embed.keys():
            t = train_embed[candi].view(encoder.hidden_size)
            e = embedded.view(encoder.hidden_size)
            temp[candi] = cosine_similarity(t, e)

        top_n = get_top_n(temp, 5)
        print()
        for e in top_n.keys():
            print("A.%s %s" % (e, pretty_printer2(train_list[e])))
            if isAnswer(e, test_answer[tk]):
                #print("top5 ", tk, e, top_n[e])
                answer5 += 1
                break
        print("-------------------------------------------------------------")
        top1 = list(top_n.keys())[0]
        if isAnswer(top1, test_answer[tk]):
            #print("top1 ", tk, top1)
            answer1 += 1

    accuracy_at_5 = answer5/total*100
    accuracy_at_1 = answer1/total*100

    print("total: %d, accuracy@5: %.4f, accuracy@1: %.4f" % (total, accuracy_at_5, accuracy_at_1))

    return accuracy_at_5, accuracy_at_1
