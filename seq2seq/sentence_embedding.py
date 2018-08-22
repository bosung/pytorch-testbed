import argparse
import evaluate as ev
from preprocess import Vocab
from features import Features, Document

from model import *
from const import *
from utils import *

EMBED_PATH = 'data/komoran_hd_2times.vec'
train_file = 'data/train_data_nv.txt'
# train_file = 'data/train_data_nv.txt'


def get_embedding(encoder, sentence, vocab, batch_size, pre_trained_embedding, features, test_features, doc, decoder=None):
    """ test different embedding method """
    # embedded = ev.get_embed(encoder, sentence, vocab, batch_size)
    # embedded = ev.get_embed_concat(encoder, decoder, sentence, vocab, batch_size)
    # embedded1 = ev.get_embed_ans_pivot(encoder, decoder, sentence, vocab, batch_size)
    # embedded2 = ev.get_embed_q_pivot(encoder, decoder, sentence, vocab, batch_size)
    # embedded = ev.get_embed_avg(encoder, vocab, sentence)
    # embedded = ev.get_embed_we_sa(encoder, decoder, sentence, vocab, batch_size, pre_trained_embedding)
    # embedded = ev.get_embed_with_ans_words(encoder, sentence, vocab, batch_size, pre_trained_embedding, features, test_features, doc, decoder=decoder)
    embedded = ev.get_hidden_vector_matrix(encoder, sentence, vocab, batch_size)
    # embedded = torch.cat((embedded1, embedded2), dim=0)
    return embedded


def build_docs(path):
    _docs = {}
    _lst = []
    lines = open(path, encoding='utf-8').read().strip().split("\n")
    for line in lines:
        q, a, num = line.split("\t")
        d = Document(num, q, a)
        _docs[num] = d
        _lst.append(d)
    return _docs, _lst


def build_test_docs(test_list, encoder, decoder, vocab, batch_size):
    _docs = {}
    _lst = []
    for tk in test_list:
        sentence = test_list[tk]
        # for batch
        sent = []
        for e in range(batch_size):
            sent.append(sentence)
        result = ev.evaluate(encoder, decoder, sent, vocab, batch_size)[0]
        ans = " ".join(result)
        d = Document(tk, sentence, ans)
        #print(sentence, ans)
        _docs[tk] = d
        _lst.append(d)
    return _docs, _lst


def evaluate_similarity(encoder, vocab, batch_size, decoder=None):
    train_list, test_list, test_answer = ev.prepare_evaluate()

    pre_trained_embedding = vocab.load_weight(EMBED_PATH)

    # tf-idf
    docs, doc_list = build_docs(train_file)
    features = Features(doc_list)

    test_docs, test_doc_list = build_test_docs(test_list, encoder, decoder, vocab, batch_size)
    test_features = Features(test_doc_list)

    # embed candidates
    train_embed = {}
    print("[INFO] encoding train %d data ..." % len(train_list))
    for d in train_list.keys():
        embedded = get_embedding(encoder, train_list[d], vocab, batch_size, pre_trained_embedding, features, test_features, docs[d])
        train_embed[d] = embedded
        return
    print("[INFO] done")
    # return

    print("[INFO] start evaluating!")
    total = len(test_list)
    answer5 = 0
    answer1 = 0
    for i, tk in enumerate(test_list.keys()):
        embedded = get_embedding(encoder, test_list[tk], vocab, batch_size, pre_trained_embedding, features, test_features, test_docs[tk], decoder=decoder)
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
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--w_embed_size', type=int, default=64)
    args = parser.parse_args()

    vocab = Vocab()
    vocab.build(train_file)
    if args.encoder:
        weight = empty_weight
    else:
        # load pre-trained embedding
        weight = vocab.load_weight(path="data/komoran_hd_2times.vec")
        # weight = vocab.load_weight(path="data/komoran_hd_2times.small")
        # weight = empty_weight

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

