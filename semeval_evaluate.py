#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals

import torch
import onmt.utils
from onmt.utils.logging import init_logger
from onmt.utils.misc import split_corpus
from onmt.model_builder import load_test_model

import onmt.inputters as inputters
import onmt.opts as opts
from onmt.utils.parse import ArgumentParser
import subprocess


def main(opt):
    logger = init_logger(opt.log_file)

    src_reader = inputters.str2reader[opt.data_type].from_opt(opt)
    fields, model, model_opt = load_test_model(opt)

    tgt_field = dict(fields)["label"]
    valid_loss = onmt.utils.loss.build_loss_compute(model, tgt_field, opt, train=False)

    src_shards = split_corpus(opt.src, 0)

    for i, src_shard in enumerate(src_shards):
        logger.info("Evaluating ...")
        _id = [line.strip().split("\t")[0] for line in src_shard[1:]]
        sent1 = [line.strip().split("\t")[1] for line in src_shard[1:]]
        sent2 = [line.strip().split("\t")[2] for line in src_shard[1:]]
        label = []
        for line in src_shard[1:]:
            token = line.strip().split("\t")[3]
            if token in ["Good", "entailment", "1", 1]:
                label.append(1)
            else:
                label.append(0)

        data = inputters.Dataset(
            fields,
            readers=([src_reader, src_reader, src_reader, src_reader]),
            data=([("id", _id), ("sent1", sent1), ("sent2", sent2), ("label", label)]),
            dirs=[None, None, None, None],
            sort_key=inputters.str2sortkey[opt.data_type],
            filter_pred=None
        )

        data_iter = inputters.OrderedIterator(
            dataset=data,
            device=torch.device("cuda", opt.gpu) if opt.gpu > -1 else torch.device("cpu"),
            batch_size=opt.batch_size,
            train=False,
            sort=False,
            sort_within_batch=False,
            shuffle=False
        )

        stats = onmt.utils.Statistics()
        ids = []
        scores = []
        with torch.no_grad():
            for batch in data_iter:
                sent1, sent2 = batch.sent1, batch.sent2

                outputs = model(sent1, sent2)
                score = torch.nn.Softmax(dim=-1)(outputs)[:, 1]

                ids.extend(batch.id)
                scores.extend(score.tolist())

                # Compute loss.
                _, batch_stats = valid_loss(batch, outputs, None)

                # Update statistics.
                stats.update(batch_stats)
        stats.print_result()
        logger.info("writing pred.txt file...")
        pred_file = "eval_utils/semeval/pred.txt"
        with open(pred_file, "w") as fw:
            for _id, score in zip(ids, scores):
                q_id = "_".join(_id.split("_")[0:2])
                ans_id = _id

                label = "true" if score > 0.5 else "false"
                fw.write("\t".join([q_id, ans_id, "0", str(score), label]))
                fw.write("\n")

        logger.info("run eval script file...")

        subprocess.run(['python2.7', 'eval_utils/semeval/ev.py',
                        'eval_utils/semeval/SemEval2017-task3-English-test-subtaskA.xml.subtaskA.relevancy', pred_file])


def _get_parser():
    parser = ArgumentParser(description='evaluate.py')

    opts.config_opts(parser)
    opts.evaluate_opts(parser)
    return parser


if __name__ == "__main__":
    parser = _get_parser()

    opt = parser.parse_args()
    main(opt)
