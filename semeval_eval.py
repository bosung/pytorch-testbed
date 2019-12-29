import torch
import subprocess
import numpy as np


def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def semeval_eval(ep, device, eval_examples, eval_dataloader, model, logger, BERT, _type):
    logger.info("***** [epoch %d] Running evaluation with official code *****" % ep)
    logger.info("  Num examples = %d", len(eval_examples))
    model.eval()
    eval_accuracy, nb_eval_example = 0, 0
    pred_data = []
    for i, batch in enumerate(eval_dataloader):
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            if BERT:
                input_ids, input_mask, segment_ids, label_ids, preprob0, preprob1 = batch
                logits, _, _ = model(input_ids, segment_ids, input_mask, labels=None)
            else:
                input_ids_a, input_ids_b, label_ids, preprob0, preprob1 = batch
                logits = model(input_ids_a, input_ids_b)

        logits = logits.detach().cpu().numpy()
        label_ids = label_ids.to('cpu').numpy()

        eval_accuracy += accuracy(logits, label_ids)
        nb_eval_example += len(label_ids)

        prob = softmax(logits[0])

        guid_token = eval_examples[i].guid.split("-")[1].split("_")

        question_id = guid_token[0] + "_" + guid_token[1]
        answer_id = eval_examples[i].guid.split("-")[1]

        rank = 0
        score = prob[1].item()
        label = "true" if score > 0.5 else "false"
        pred_data.append([question_id, answer_id, rank, score, label])

    eval_accuracy = eval_accuracy / nb_eval_example

    pred_file = "semeval/pred_{}_{}.txt".format(_type, str(ep))
    logger.info("***** [epoch %d] write file: %s *****" % (ep, pred_file))
    with open(pred_file, "w") as f:
        for d in pred_data:
            f.write("\t".join([str(e) for e in d]) + "\n")

    logger.info("***** [epoch %d] Done " % ep)

    if _type == "test":
        subprocess.run(['python2.7', 'semeval/ev.py',
                            'semeval/SemEval2017-task3-English-test-subtaskA.xml.subtaskA.relevancy', pred_file])
    else:
        subprocess.run(['python2.7', 'semeval/ev.py',
                        'semeval/SemEval2016-Task3-CQA-QL-dev-subtaskA.xml.subtaskA.relevancy', pred_file])

    return eval_accuracy
