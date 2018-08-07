import time
import math

import torch

from const import *

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np


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


def pretty_printer(data):
    return [x.split("/")[0] for x in data]


def pretty_printer2(data):
    return ' '.join([x.split("/")[0] for x in data.split(" ")][:MAX_LENGTH])


def showPlot(points, title):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    plt.savefig(title)


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

