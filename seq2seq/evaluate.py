import preprocess as prep
from const import *

# evaluate

train_list, test_list, test_answer = prep.prepare_evaluate()


def get_embed(encoder, sentence, vocab, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = prep.tensorFromSentence(vocab, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.init_hidden()

        # need for attention. not now
        # encoder_outputs = torch.zeros(max_length, 1, encoder.hidden_dim, device=device)

        for i in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[i], encoder_hidden)

        # encoder_hidden -> sentence embedding
        return encoder_hidden


def evaluate(encoder, vocab):
    # embed candidates
    train_embed = {}
    print("[INFO] encoding train %d data ..." % len(train_list))
    for d in train_list.keys():
        embedded = get_embed(encoder, train_list[d], vocab)
        train_embed[d] = embedded
    print("[INFO] done")

    print("[INFO] start evaluating!")
    total = len(test_list)
    answer5 = 0
    answer1 = 0
    for tk in test_list.keys():
        embedded = get_embed(encoder, test_list[tk], vocab)
        temp = {}
        for candi in train_embed.keys():
            t = train_embed[candi].view(encoder.hidden_dim)
            e = embedded.view(encoder.hidden_dim)
            temp[candi] = cosine_similarity(t, e)
            #print("[INFO] test: %s train: %s" % (tk, candi))

        top_n = get_top_n(temp, 5)
        for e in top_n.keys():
            if e == test_answer[tk]:
                answer5 += 1
                break

        top_n = get_top_n(temp, 1)
        for e in top_n.keys():
            if e == test_answer[tk]:
                answer1 += 1
                break

    print("total: %d, accuracy@5: %.4f, accuracy@1: %.4f" % (total, answer5/total*100, answer1/total*100))
