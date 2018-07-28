import preprocess as prep
from const import *
#from utils import cosine_similarity
from utils import *

from nltk.translate.bleu_score import sentence_bleu


def prepare_evaluate():
    train_data = {}
    test_data = {}
    test_answer = {}

    lines = open('data/train_list.txt', 'r').read().strip().split('\n')
    for l in lines:
        q, num = l.split('\t')
        train_data[num] = q

    lines = open('data/test_list.txt', 'r').read().strip().split('\n')
    for l in lines:
        q, num, answer = l.split('\t')
        test_data[num] = q
        test_answer[num] = answer

    return train_data, test_data, test_answer


def get_embed(encoder, sentence, vocab, batch_size, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = prep.tensorFromSentenceBatchWithPadding(vocab, [sentence])

        # because of batch, need expansion for input tensor
        temp = input_tensor
        for _ in range(batch_size-1):
            temp = torch.cat((temp, input_tensor), 0)
        input_tensor = temp

        encoder_hidden = encoder.init_hidden(max_length)

        # need for attention. not now
        # encoder_outputs = torch.zeros(max_length, 1, encoder.hidden_dim, device=device)

        encoder_output, encoder_hidden = encoder(input_tensor, encoder_hidden)

        # encoder_hidden -> sentence embedding
        return encoder_hidden[0][0].view(1, 1, -1)


def isAnswer(answer, predict):
    if answer == predict:
        return True
    elif ((answer == 51 or answer == 308) and (predict == 51 or predict == 308)): return True
    elif ((answer == 229 or answer == 46) and (predict == 229 or predict == 46)): return True
    elif ((answer == 271 or answer == 47) and (predict == 271 or predict == 47)): return True
    elif ((answer == 24 or answer == 200) and (predict == 24 or predict == 200)): return True
    elif ((answer == 25 or answer == 201) and (predict == 25 or predict == 201)): return True
    elif ((answer == 274 or answer == 225) and (predict == 274 or predict == 225)): return True
    elif ((answer == 20 or answer == 175) and (predict == 20 or predict == 175)): return True
    elif ((answer == 56 or answer == 350) and (predict == 56 or predict == 350)): return True
    elif ((answer == 13 or answer == 100) and (predict == 13 or predict == 100)): return True
    elif ((answer == 404 or answer == 405) and (predict == 404 or predict == 405)): return True
    elif ((answer == 61 or answer == 367) and (predict == 61 or predict == 367)): return True
    elif ((answer == 17 or answer == 148) and (predict == 17 or predict == 148)): return True
    elif ((answer == 533 or answer == 554) and (predict == 533 or predict == 554)): return True
    elif ((answer == 14 or answer == 124) and (predict == 14 or predict == 124)): return True
    elif ((answer == 444 or answer == 529 or answer == 531) and (predict == 444 or predict == 529 or predict == 531)): return True
    else: return False


def evaluate_similarity(encoder, vocab, batch_size):
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
        embedded = get_embed(encoder, test_list[tk], vocab, batch_size)
        temp = {}
        for candi in train_embed.keys():
            t = train_embed[candi].view(encoder.hidden_dim)
            e = embedded.view(encoder.hidden_dim)
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

        encoder_hidden = encoder.init_hidden(max_length)

        encoder_outputs = torch.zeros(max_length, encoder.hidden_dim, device=device)

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
    print("Average BLEU score: %.2f" % (total_bleu_score/len(result_batch)))


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
            t = train_embed[candi].view(encoder.hidden_dim)
            e = embedded.view(encoder.hidden_dim)
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
