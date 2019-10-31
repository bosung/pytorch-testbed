import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import math


def sampling_ploting(t_prob, ep):
    x = np.arange(0.0, 1.001, 0.001)
    y = np.zeros(1001)
    for p in t_prob:
        y[int(p*1000)] += 1
    # y = np.exp(y)/np.sum(np.exp(y))
    plt.plot(x, y)
    plt.grid(which='major', alpha=0.1)
    major_ticks = np.arange(0, 1.1, 0.1)
    plt.xticks(major_ticks)
    plt.xlabel('p(y|x)')
    plt.ylabel('# of examples ')
    probs = np.array(t_prob)
    plt.vlines(probs.mean(), ymin=0, ymax=y.max(), label=("μ=%.2f" % probs.mean()), colors='r')
    plt.vlines(0.5, ymin=0, ymax=y.max(), label='0.5', colors='br')
    plt.legend(loc='center right')
    # plt.show()
    plt.savefig("ep-{}.png".format(str(ep)))


def output_ploting(t_neg, t_pos, ep):
    x = np.arange(0.0, 1.001, 0.001)
    y_p = np.zeros(1001)
    y_n = np.zeros(1001)
    for p in t_pos:
        y_p[int(p*1000)] += 1
    for p in t_neg:
        y_n[int(p*1000)] += 1
    # y = np.exp(y)/np.sum(np.exp(y))
    plt.plot(x, y_n)
    plt.plot(x, y_p)
    plt.grid(which='major', alpha=0.1)
    major_ticks = np.arange(0, 1.1, 0.1)
    plt.xticks(major_ticks)
    plt.xlabel('p(y|x)')
    plt.ylabel('# of examples ')
    probs_p = np.array(t_pos)
    probs_n = np.array(t_neg)
    height_max = max(y_n.max(), y_p.max())
    plt.vlines(probs_n.mean(), ymin=0, ymax=height_max, label=("μ=%.2f" % probs_n.mean()), colors='b')
    plt.vlines(probs_p.mean(), ymin=0, ymax=height_max, label=("μ=%.2f" % probs_p.mean()), colors='orange')
    plt.vlines(0.5, ymin=0, ymax=height_max, label='0.5', colors='r')
    plt.legend(loc='center right')
    # plt.show()
    plt.savefig("sep-ep-{}.png".format(str(ep)))


if __name__ == "__main__":
    sampling_ploting([0.1, 0.2, 0.3, 0.2, 0.2, 0.3, 0.4], 1)
