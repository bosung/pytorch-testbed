import matplotlib.pyplot as plt
import numpy as np
# import scipy.stats as stats
import math


def plot_samples(t_prob, ep):
    x = np.arange(0.0, 1.001, 0.001)
    y = np.zeros(1001)
    for p in t_prob:
        y[int(p*1000)] += 1
    y = y / np.sum(y)
    plt.plot(x, y)
    plt.grid(which='major', alpha=0.1)
    major_ticks = np.arange(0, 1.1, 0.1)
    plt.xticks(major_ticks)
    plt.xlabel(r'$\widehat{y}=F(x)$')
    plt.ylabel(r'$p(\widehat{y}|x)$')
    probs = np.array(t_prob)
    plt.vlines(probs.mean(), ymin=0, ymax=y.max(), label=("μ=%.2f" % probs.mean()), colors='r')
    plt.vlines(0.5, ymin=0, ymax=y.max(), label='0.5', colors='br')
    plt.legend(loc='center right')
    # plt.show()
    plt.savefig("ep-{}.png".format(str(ep)))


def plot_out_dist(t_neg, t_pos, ep):
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


def plot_vector_bar(v, u, class_names):
    """ ref: https://matplotlib.org/3.1.1/gallery/lines_bars_and_markers/barchart.html """
    fig, ax = plt.subplots()
    # plt.imshow(v, interpolation='nearest', cmap=plt.cm.Blues)
    # plt.title("title")
    # plt.colorbar()

    x = np.arange(len(class_names))
    width = 0.35

    # tick_marks = np.arange(len(class_names))
    # plt.xticks(tick_marks, class_names, rotation=45)

    v = np.around(v, decimals=2)
    u = np.around(u, decimals=2)
    rects = ax.bar(x - width/2, v, width, label='true')
    rects2 = ax.bar(x + width/2, u, width, label='prediction')

    def annotate(rects):
        for i, rect in enumerate(rects):
            height = rect.get_height()
            plt.annotate(height,
                         xy=(rect.get_x() + rect.get_width() / 2, height),
                         ha='center', va='bottom')

    annotate(rects)
    annotate(rects2)

    fig.tight_layout()
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=40)
    ax.legend()
    # plt.ylabel(r'$p(y|x)$')
    ax.set_ylabel(r'$p(y|x)$')
    # plt.xlabel('class')
    ax.set_xlabel('class')
    # plt.show()
    return fig


if __name__ == "__main__":
    # plot_samples([0.1, 0.2, 0.3, 0.2, 0.2, 0.3, 0.4], 1)
    plot_vector_bar([0.1, 0.2, 0.3, 0.2, 0.2, 0.3, 0.4],
                    [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
                    ["a", "b", "c", "d", "e", "f", "g"])
