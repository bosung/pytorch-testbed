import preprocess as prep
from const import *
from utils import cosine_similarity
from utils import get_top_n


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


def evaluate(encoder, vocab, batch_size):
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
    for tk in test_list.keys():
        embedded = get_embed(encoder, test_list[tk], vocab, batch_size)
        temp = {}
        for candi in train_embed.keys():
            t = train_embed[candi].view(encoder.hidden_dim)
            e = embedded.view(encoder.hidden_dim)
            temp[candi] = cosine_similarity(t, e)

        top_n = get_top_n(temp, 5)
        for e in top_n.keys():
            if isAnswer(e, test_answer[tk]):
                #print("top5 ", tk, e, top_n[e])
                answer5 += 1
                break

        top1 = list(top_n.keys())[0]
        if isAnswer(top1, test_answer[tk]):
            #print("top1 ", tk, top1)
            answer1 += 1

    accuracy_at_5 = answer5/total*100
    accuracy_at_1 = answer1/total*100

    print("total: %d, accuracy@5: %.4f, accuracy@1: %.4f" % (total, accuracy_at_5, accuracy_at_1))

    return accuracy_at_5, accuracy_at_1
