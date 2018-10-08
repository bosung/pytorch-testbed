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
    # embedded = ev.get_word_embed_avg(vocab, sentence, pre_trained_embedding)
    # embedded = ev.get_embed_concat(encoder, decoder, sentence, vocab, batch_size)
    # embedded1 = ev.get_embed_ans_pivot(encoder, decoder, sentence, vocab, batch_size)
    # embedded2 = ev.get_embed_q_pivot(encoder, decoder, sentence, vocab, batch_size)
    # embedded = ev.get_embed_avg(encoder, vocab, sentence)
    # embedded = ev.get_embed_we_sa(encoder, decoder, sentence, vocab, batch_size, pre_trained_embedding)
    # embedded = ev.get_embed_with_ans_words(encoder, sentence, vocab, batch_size, pre_trained_embedding, features, test_features, doc, decoder=decoder)
    # embedded = ev.get_hidden_vector_matrix(encoder, sentence, vocab, batch_size)
    # embedded = torch.cat((embedded1, embedded2), dim=0)
    embedded = ev.get_attn_hidden_avg(encoder, decoder, sentence, vocab, batch_size)
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
    train_list, _, test_list, test_answer = ev.prepare_evaluate()

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
        # embedded = get_embedding(encoder, train_list[d], vocab, batch_size, pre_trained_embedding, features, test_features, docs[d])
        embedded = get_embedding(encoder, docs[d].q, vocab, batch_size, pre_trained_embedding, features, test_features, docs[d], decoder=decoder)
        train_embed[d] = embedded
    print("[INFO] done")
    # return

    print("[INFO] start evaluating!")
    total = len(test_list)
    answer5 = 0
    answer1 = 0
    for i, tk in enumerate(test_list.keys()):
        # embedded = get_embedding(encoder, test_list[tk], vocab, batch_size, pre_trained_embedding, features, test_features, test_docs[tk], decoder=decoder)
        embedded = get_embedding(encoder, test_docs[tk].q, vocab, batch_size, pre_trained_embedding, features, test_features, test_docs[tk], decoder=decoder)
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


def get_top_n_idx(tensor1d, n):
    d = {}
    for i, x in enumerate(tensor1d.data):
        d[i] = float(x)
    result = {}
    num = 0
    for key, value in reversed(sorted(d.items(), key=lambda i: (i[1], i[0]))):
        result[key] = value
        num += 1
        if num == n:
            break
    return list(result.keys())


def get_hirel_n_ans_avg(h_bar, h_tilda, n):
    associate_matrix = torch.matmul(h_bar, h_tilda.transpose(0, 1))
    reduced = torch.sum(associate_matrix, 0)
    high_rel_idx = get_top_n_idx(reduced, n)

    high_rel_answer = h_tilda[high_rel_idx[0]].unsqueeze(0)
    for i in range(1, len(high_rel_idx)):
        high_rel_answer = torch.cat((high_rel_answer, h_tilda[high_rel_idx[i]].unsqueeze(0)), 0)

    return torch.mean(high_rel_answer, 0)


def eval_sim_lc(encoder, vocab, batch_size, pre_trained_embedding, decoder=None, alpha=0.9):
    train_list, _, test_list, test_answer = ev.prepare_evaluate()

    # tf-idf
    docs, doc_list = build_docs(train_file)
    features = Features(doc_list)

    test_docs, test_doc_list = build_test_docs(test_list, encoder, decoder, vocab, batch_size)
    test_features = Features(test_doc_list)

    # embed candidates
    train_q_embed = {}
    train_a_embed = {}
    print("[INFO] encoding train %d data ..." % len(train_list))
    for d in docs.keys():
        # TODO
        h_bar, h_tilda = ev.get_hiddens(encoder, decoder, docs[d].q, vocab, batch_size)
        # train_a_embed[d] = get_hirel_n_ans_avg(h_bar, h_tilda, 3)
        # h_bar.size() = (15, 300)
        train_q_embed[d] = torch.mean(h_bar, 0)
        train_a_embed[d] = torch.mean(h_tilda, 0)
    print("[INFO] done")

    print("[INFO] start evaluating!")
    print("==================>", alpha)
    total = len(test_list)
    answer5 = 0
    answer1 = 0
    for i, tk in enumerate(test_docs.keys()):
        h_bar_, h_tilda_ = ev.get_hiddens(encoder, decoder, test_docs[tk].q, vocab, batch_size)

        # a_embed = get_hirel_n_ans_avg(h_bar_, h_tilda_, 3)
        a_embed = torch.mean(h_tilda_, 0)
        q_embed = torch.mean(h_bar_, 0)

        # cacluate score
        temp_q = {}
        temp_a = {}
        temp = {}
        for candi in train_q_embed.keys():
            # question part
            tq = train_q_embed[candi].view(-1)
            eq = q_embed.view(-1)
            temp_q[candi] = cosine_similarity(tq, eq)
            # answer part
            ta = train_a_embed[candi].view(-1)
            ea = a_embed.view(-1)
            temp_a[candi] = cosine_similarity(ta, ea)

            temp[candi] = alpha*temp_q[candi] + (1-alpha)*temp_a[candi]
            # temp[candi] = temp_q[candi]

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
    """
    print("-------------------> self-attention")
    # embed candidates
    train_q_embed = {}
    train_a_embed = {}
    print("[INFO] encoding train %d data ..." % len(train_list))
    for d in docs.keys():
        train_q_embed[d] = ev.get_word_embed_avg_sa(vocab, docs[d].q, pre_trained_embedding)
        train_a_embed[d] = ev.get_word_embed_avg_sa(vocab, docs[d].ans, pre_trained_embedding)
        # train_q_embed[d], train_a_embed[d] = ev.get_ende_hidden(encoder, decoder, docs[d].q, vocab, batch_size)
    print("[INFO] done")
    # return

    print("[INFO] start evaluating!")
    print("==================>", alpha)
    total = len(test_list)
    answer5 = 0
    answer1 = 0
    for i, tk in enumerate(test_docs.keys()):
        q_embed = ev.get_word_embed_avg_sa(vocab, test_docs[tk].q, pre_trained_embedding)
        a_embed = ev.get_de_out_embed_sa(encoder, decoder, vocab, test_docs[tk].q, 40, pre_trained_embedding)
        # q_embed, a_embed = ev.get_ende_hidden(encoder, decoder, test_docs[tk].q, vocab, batch_size)
        # cacluate score
        temp_q = {}
        temp_a = {}
        temp = {}
        for candi in train_q_embed.keys():
            # question part
            tq = train_q_embed[candi].view(-1)
            eq = q_embed.view(-1)
            temp_q[candi] = cosine_similarity(tq, eq)
            # answer part
            ta = train_a_embed[candi].view(-1)
            ea = a_embed.view(-1)
            temp_a[candi] = cosine_similarity(ta, ea)

            temp[candi] = alpha*temp_q[candi] + (1-alpha)*temp_a[candi]
            # temp[candi] = temp_q[candi]

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
    """

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder', help='load exisiting model')
    parser.add_argument('--decoder', help='load exisiting model')
    parser.add_argument('--batch_size', type=int, default=40)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--w_embed_size', type=int, default=64)
    parser.add_argument('--pre_trained_embed', choices=['y', 'n'], default='n')
    args = parser.parse_args()

    vocab = Vocab()
    vocab.build(train_file)

    batch_size = args.batch_size
    hidden_size = args.hidden_size
    w_embed_size = args.w_embed_size

    if args.pre_trained_embed == 'n':
        encoder = Encoder(vocab.n_words, w_embed_size, hidden_size, batch_size).to(device)
        decoder = AttentionDecoder(vocab.n_words, w_embed_size, hidden_size, batch_size).to(device)
        # decoder = Decoder(vocab.n_words, w_embed_size, hidden_size, batch_size).to(device)
    else:
        # load pre-trained embedding
        weight = vocab.load_weight(path="data/komoran_hd_2times.vec")
        encoder = Encoder(vocab.n_words, w_embed_size, hidden_size, batch_size, weight).to(device)
        decoder = AttentionDecoder(vocab.n_words, w_embed_size, hidden_size, batch_size, weight).to(device)
        # decoder = Decoder(vocab.n_words, w_embed_size, hidden_size, batch_size, weight).to(device)

    if args.encoder:
        encoder.load_state_dict(torch.load(args.encoder))
        print("[INFO] load encoder with %s" % args.encoder)
    if args.decoder:
        decoder.load_state_dict(torch.load(args.decoder))
        print("[INFO] load decoder with %s" % args.decoder)

    # evaluate_similarity(encoder, vocab, batch_size, decoder=decoder)
    pre_trained_embedding = vocab.load_weight(EMBED_PATH)
    eval_sim_lc(encoder, vocab, batch_size, pre_trained_embedding, decoder=decoder, alpha=1)
    eval_sim_lc(encoder, vocab, batch_size, pre_trained_embedding, decoder=decoder, alpha=0.95)
    eval_sim_lc(encoder, vocab, batch_size, pre_trained_embedding, decoder=decoder, alpha=0.9)
    eval_sim_lc(encoder, vocab, batch_size, pre_trained_embedding, decoder=decoder, alpha=0.85)
    eval_sim_lc(encoder, vocab, batch_size, pre_trained_embedding, decoder=decoder, alpha=0.8)
    eval_sim_lc(encoder, vocab, batch_size, pre_trained_embedding, decoder=decoder, alpha=0.7)
    #eval_sim_lc(encoder, vocab, batch_size, pre_trained_embedding, decoder=decoder, alpha=0.6)

    """
    train_list, test_list, test_answer = ev.prepare_evaluate()

    # tf-idf
    docs, doc_list = build_docs(train_file)
    test_docs, test_doc_list = build_test_docs(test_list, encoder, decoder, vocab, batch_size)

    for i, tk in enumerate(test_docs):
        sent = []
        for e in range(batch_size):
            sent.append(test_docs[tk].q)
        result = ev.evaluate(encoder, decoder, sent, vocab, batch_size)
        print('Q.%s %s' % (tk, pretty_printer2(test_docs[tk].q)))
        if test_answer[tk] == '26':
            print('')
            continue
        print('=', pretty_printer2(docs[test_answer[tk]].q))
        #output_words = evaluate(encoder, decoder, pair[0], vocab)
        out = result[0]
        output_sentence = ' '.join(pretty_printer(out))
        print('<', output_sentence)
        print('')
    """
