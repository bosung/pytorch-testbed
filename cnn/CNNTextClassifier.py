import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F


class CNNTextClassifier(nn.Module):
    """Implementaion for the paper Yoon Kim, Convolutional Neural Networks for 
    Sentence Classification, EMNLP 2017
    """

    def __init__(self, in_channels, out_channels, word_embed_dim, drop_out):
        super(CNNTextClassifier, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.word_embed_dim = word_embed_dim
        self.MAX_SENTENCE_SIZE = 80

        # convolution operations with 
        # filter window h = 3
        self.conv1 = nn.Conv2d(1, 100, (3, self.word_embed_dim))
        # filter window h = 4
        self.conv2 = nn.Conv2d(1, 100, (4, self.word_embed_dim))
        # filter window h = 5
        self.conv3 = nn.Conv2d(1, 100, (5, self.word_embed_dim))

        self.drop_out = nn.Dropout(p=drop_out)
        self.linear = nn.Linear(300, self.out_channels)


    def forward(self, inputs):
        inputs = self.padding(inputs)

        feature_map1 = F.relu(self.conv1(inputs))
        feature_map2 = F.relu(self.conv2(inputs))
        feature_map3 = F.relu(self.conv3(inputs))

        out1 = F.max_pool2d(feature_map1, (self.MAX_SENTENCE_SIZE-2, 1)).view(-1)
        out2 = F.max_pool2d(feature_map2, (self.MAX_SENTENCE_SIZE-3, 1)).view(-1)
        out3 = F.max_pool2d(feature_map3, (self.MAX_SENTENCE_SIZE-4, 1)).view(-1)

        result = torch.cat([out1, out2, out3], dim=0)
        tag_score = self.linear(result)
        ret = F.log_softmax(tag_score, dim=0).view(1, -1)
        return ret


    def padding(self, inputs):
        zeros = torch.zeros(1, 1, self.MAX_SENTENCE_SIZE-inputs.size(2), self.word_embed_dim)
        return autograd.Variable(torch.cat([inputs, zeros], dim=2))

