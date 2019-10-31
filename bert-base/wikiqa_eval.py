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
    logger.info("\taccuracy\tprecision\trecall\t\tF1\t\tMRR\t\tMAP")
    logger.info("\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f" % (
        eval_accuracy, results[3], results[4], results[5], results[7], results[8]))

    result_log = "%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f" % (
        eval_accuracy, results[3], results[4], results[5], results[7], results[8])
    return results[5], result_log  # return f1 score and log


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
    preqpos = data[0][0]
    q_list = []
    temp = []
    for parts in data:
        qpos, label, scr = parts[0], int(parts[3]), float(parts[4])
        if qpos != preqpos:
            q_list.append(temp)
            temp = []
            preqpos = qpos
        temp.append([qpos, label, scr])
    q_list.append(temp)

    def sort_by_score(val):
        return val[2]

    temp_mrr = 0
    total_map = 0
    valid_mrr_cnt = 0
    valid_map_cnt = 0
    for q in q_list:
        q.sort(key=sort_by_score, reverse=True)
        # for MRR
        for i, elem in enumerate(q, 1):
            if elem[1] == 1:
                temp_mrr += (1/i)
                valid_mrr_cnt += 1
                break
        # for MAP
        positive = 0
        temp_map = 0
        for i, elem in enumerate(q, 1):
            if positive == 0 and elem[1] == 1:
                valid_map_cnt += 1
            if elem[1] == 1:
                positive += 1
                temp_map += (positive/i)
        if positive > 0:
            ap = temp_map / positive
            total_map += ap
    mrr = temp_mrr/valid_mrr_cnt
    map = total_map/valid_map_cnt

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
    qprec = match_cnt / pred_cnt if pred_cnt != 0 else 0
    qreca = match_cnt / ref_cnt

    qmatch_cnt, qcnt = 0.0, 0.0
    for r, pidx in zip(qref, qpred_idx):
        qcnt += 1.0
        if r == 1 and pred[pidx] >= thre and ref[pidx] == 1:
            qmatch_cnt += 1.0
        elif r == 0 and pred[pidx] < thre:
            qmatch_cnt += 1.0
    qacc = qmatch_cnt / qcnt

    return [prec, reca, 2.0 * prec * reca / (prec + reca) if (prec + reca) != 0 else 0,
            qprec, qreca, 2.0 * qprec * qreca / (qprec + qreca) if (qprec + qreca) != 0 else 0,
            qacc, mrr, map]
