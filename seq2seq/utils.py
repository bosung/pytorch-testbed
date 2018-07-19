import time
import math

import torch


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def cosine_similarity(a, b):
    # a and b are tensor
    return torch.dot(a, b) / (torch.norm(a) * torch.norm(b))


def get_top_n(data, n):
    sorted_dict = {}
    num = 0
    for key, value in reversed(sorted(data.items(), key=lambda i: (i[1], i[0]))):
        sorted_dict[key] = value
        num += 1
        if num == n:
            break
    return sorted_dict

