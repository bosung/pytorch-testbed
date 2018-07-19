import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SOS_token = 0
EOS_token = 1
UNK_token = 2

MAX_LENGTH = 20
