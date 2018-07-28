
import preprocess as prep

from preprocess import Vocab

TRAIN_PATH = 'data/cqa_train.txt'

if __name__=="__main__":
    vocab = Vocab()
    vocab.build(TRAIN_PATH)
    # torch.tensor([2764, 64])
    pre_trained_embedding = vocab.load_weight()

    train_data = prep.read_train_data(TRAIN_PATH)
    sentences = [prep.tensorFromSentence(vocab, s[0]) for s in train_data]

    for sentence in sentences:
        # make sentence matrix
        x = pre_trained_embedding[sentence[0]].view(1, -1)
        for i in sentence[1:]:
            x = torch.cat((x, pre_trained_embedding[i].view(1, -1)), 0)

        # sent_matrix.size() = torch.Size([n, d])
        attn_matrix = torch.matmul(x, x.transpose(0, 1))
        self_attn_matrix = torch.matnul(attn_matrix, x)

        # represent sentece by averaging matrix
        applied_sent = torch.mean(self_attn_matrix, 0)
        break


