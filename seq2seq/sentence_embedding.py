import argparse
import evaluate as ev
from preprocess import Vocab

from model import *
from const import *
from utils import *


def get_embedding(encoder, decoder, sentence, vocab, batch_size):
    """ test different embedding method """
    #embedded = ev.get_embed(encoder, sentence, vocab, batch_size)
    #embedded = ev.get_embed_concat(encoder, decoder, sentence, vocab, batch_size)
    #embedded = ev.get_embed_ans_pivot(encoder, decoder, sentence, vocab, batch_size)
    embedded = ev.get_embed_q_pivot(encoder, decoder, sentence, vocab, batch_size)
    return embedded


def evaluate_similarity(encoder, vocab, batch_size, decoder=None):
    train_list, test_list, test_answer = ev.prepare_evaluate()

    # embed candidates
    train_embed = {}
    print("[INFO] encoding train %d data ..." % len(train_list))
    for d in train_list.keys():
        embedded = get_embedding(encoder, decoder, train_list[d], vocab, batch_size)
        train_embed[d] = embedded
    print("[INFO] done")

    print("[INFO] start evaluating!")
    total = len(test_list)
    answer5 = 0
    answer1 = 0
    for i, tk in enumerate(test_list.keys()):
        embedded = get_embedding(encoder, decoder, test_list[tk], vocab, batch_size)

        # cacluate score
        temp = {}
        for candi in train_embed.keys():
            #t = train_embed[candi].view(encoder.hidden_dim)
            t = train_embed[candi].view(-1)
            #e = embedded.view(encoder.hidden_dim)
            e = embedded.view(-1)
            temp[candi] = cosine_similarity(t, e)

        # sort by cos_sim
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


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder', help='load exisiting model')
    parser.add_argument('--decoder', help='load exisiting model')
    parser.add_argument('--batch_size', type=int, default=40)
    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('--w_embed_size', type=int, default=64)
    args = parser.parse_args()

    train_file = 'data/cqa_train_komoran.txt'

    vocab = Vocab()
    vocab.build(train_file)
    if args.encoder:
        weight = empty_weight
    else:
        # load pre-trained embedding
        weight = vocab.load_weight(path="data/komoran_hd_2times.vec")
        #weight = vocab.load_weight(path="data/komoran_hd_2times.small")

    batch_size = args.batch_size

    hidden_size = args.hidden_size
    w_embed_size = args.w_embed_size

    encoder = Encoder(vocab.n_words, w_embed_size, hidden_size, batch_size, weight).to(device)
    decoder = Decoder(vocab.n_words, w_embed_size, hidden_size, batch_size, weight).to(device)

    if args.encoder:
        encoder.load_state_dict(torch.load(args.encoder))
        print("[INFO] load encoder with %s" % args.encoder)
    if args.decoder:
        decoder.load_state_dict(torch.load(args.decoder))
        print("[INFO] load decoder with %s" % args.decoder)


    evaluate_similarity(encoder, vocab, batch_size, decoder=decoder)

