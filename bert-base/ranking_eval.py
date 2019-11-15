
def sort_by_score(val):
    return val[1]


def ranking_eval(labels, logits, ids):
    n_exam = len(ids)

    t_mrr = 0
    t_r1 = 0
    t_r10 = 0
    t_r50 = 0
    temp = []
    pre_eid = ids[0].split("_")[0]
    for i, e in enumerate(ids):
        eid, rid = e.split("_")
        if pre_eid != eid:  # new example
            assert len(temp) == 100  # candidates num
            temp.sort(key=sort_by_score, reverse=True)
            for j, t in enumerate(temp, 1):
                if t[2] == 1:  # if answer
                    t_mrr += 1/j
                    if j <= 50:
                        t_r50 += 1
                    if j <= 10:
                        t_r10 += 1
                    if j <= 1:
                        t_r1 += 1
            temp = []
        temp.append([rid, logits[i][1], labels[i]])
        pre_eid = eid

    assert len(temp) == 100  # candidates num
    temp.sort(key=sort_by_score, reverse=True)
    for j, t in enumerate(temp, 1):
        if t[2] == 1:  # if answer
            t_mrr += 1/j
            if j <= 50:
                t_r50 += 1
            if j <= 10:
                t_r10 += 1
            if j <= 1:
                t_r1 += 1

    return t_r1/n_exam, t_r10/n_exam, t_r50/n_exam, t_mrr/n_exam
