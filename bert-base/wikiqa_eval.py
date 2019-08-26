import torch
import numpy as np


def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def wikiqa_eval(ep, device, eval_examples, eval_dataloader, model, logger):
    logger.info("***** [epoch %d] Running evaluation with official code *****" % ep)
    logger.info("  Num examples = %d", len(eval_examples))
    model.eval()
    eval_accuracy, nb_eval_example = 0, 0
    data = []
    for i, batch in enumerate(eval_dataloader):
        # input_ids, input_mask, segment_ids, label_ids, _, _ = batch
        input_ids = batch[0].to(device)
        input_mask = batch[1].to(device)
        segment_ids = batch[2].to(device)
        label_ids = batch[3].to(device)

        with torch.no_grad():
            logits = model(input_ids, segment_ids, input_mask, labels=None)

        logits = logits.detach().cpu().numpy()
        label_ids = label_ids.to('cpu').numpy()

        eval_accuracy += accuracy(logits, label_ids)
        nb_eval_example += input_ids.size(0)

        logits2 = softmax(logits[0])
        data.append([eval_examples[i].guid, "-", "-", label_ids[0], logits2[1]])

    eval_accuracy = eval_accuracy / nb_eval_example
    results = get_prf(data, thre=0.11)
    logger.info("\tWikiQA Question Triggering:")
    logger.info("\taccuracy\tprecision\trecall\t\tF1\t\tMRR")
    logger.info("\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f" % (
        eval_accuracy, results[3], results[4], results[5], results[6]))
    return results[5]  # return f1 score


def get_prf(data, thre=0.11):
    """
    get Q-A level and Q level precision, recall, fmeasure
    """
    ref, pred = [], []
    qref, qpred_idx = [], []
    preqpos = ""
    qflag = False

    for parts in data:
        qpos, l = parts[0], int(parts[3])  # parts[0] = question id / # parts[3] = label
        if qpos != preqpos and preqpos != "":  # per question
            if qflag:  # if qflag is true, then label is 1 (=> answer)
                qref.append(1)  # qref = [0, 0, 0, 1, 0 , ..] / ref means answer
            else:
                qref.append(0)
            qflag = False
        preqpos = qpos
        ref.append(l)  # ref = [1, 0, 0, 0, 1, ...] => total answers
        if l == 1: qflag = True
    # for last line
    if qflag:
        qref.append(1)
    else:
        qref.append(0)

    preqpos = ""
    maxval = 0.0
    maxidx = -1

    for i, parts in enumerate(data):
        qpos, scr = parts[0], float(parts[4])  # question id, scr means score. label 1's score.
        if qpos != preqpos and preqpos != "":
            qpred_idx.append(maxidx)  # qpred_idx => max idx per question
            maxval = 0.0
            maxidx = -1
        preqpos = qpos
        if scr >= thre:
            pred.append(1)  # pred = [1, 0, 0, ...]
        else:
            pred.append(0)
        if scr > maxval:
            maxidx = i
            maxval = scr
    qpred_idx.append(maxidx)

    ########################
    # calculate MRR
    ########################
    preqpos = ""
    q_list = []
    temp = []
    for parts in data:
        qpos, label, scr = parts[0], int(parts[3]), float(parts[4])
        if qpos != preqpos and preqpos != "":
            q_list.append(temp)
            temp = []
        temp.append([qpos, label, scr])
    q_list.append(temp)

    def sort_by_score(val):
        return val[2]

    temp_mrr = 0
    for q in q_list:
        q.sort(key=sort_by_score, reverse=True)
        for i, elem in enumerate(data, 1):
            if elem[1] == 1:
                temp_mrr += (1/i)
    mrr = temp_mrr/len(q_list)

    match_cnt, ref_cnt, pred_cnt = 0.0, 0.0, 0.0
    for r, p in zip(ref, pred):
        if r == 1: ref_cnt += 1.0
        if p == 1: pred_cnt += 1.0
        if r == 1 and p == 1: match_cnt += 1.0
    prec = match_cnt / pred_cnt if pred_cnt != 0 else 0
    reca = match_cnt / ref_cnt

    match_cnt, ref_cnt, pred_cnt = 0.0, 0.0, 0.0
    for r, pidx in zip(qref, qpred_idx):
        if r == 1: ref_cnt += 1.0
        if pred[pidx] >= thre: pred_cnt += 1.0
        if r == 1 and pred[pidx] >= thre and ref[pidx] == 1: match_cnt += 1.0
    qprec, qreca = match_cnt / pred_cnt, match_cnt / ref_cnt

    qmatch_cnt, qcnt = 0.0, 0.0
    for r, pidx in zip(qref, qpred_idx):
        qcnt += 1.0
        if r == 1 and pred[pidx] >= thre and ref[pidx] == 1:
            qmatch_cnt += 1.0
        elif r == 0 and pred[pidx] < thre:
            qmatch_cnt += 1.0
    qacc = qmatch_cnt / qcnt

    return [prec, reca, 2.0 * prec * reca / (prec + reca), qprec, qreca, 2.0 * qprec * qreca / (qprec + qreca), qacc, mrr]
