#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from itertools import repeat

import torch
import onmt.utils
from onmt.utils.logging import init_logger
from onmt.utils.misc import split_corpus
# from onmt.translate.translator import build_translator
from onmt.model_builder import load_test_model

import onmt.inputters as inputters
import onmt.opts as opts
from onmt.utils.parse import ArgumentParser


def main(opt):
    # ArgumentParser.validate_translate_opts(opt)
    logger = init_logger(opt.log_file)

    # translator = build_translator(opt, report_score=True)

    src_reader = inputters.str2reader[opt.data_type].from_opt(opt)
    # load_test_model = onmt.model_builder.load_test_model
    fields, model, model_opt = load_test_model(opt)

    tgt_field = dict(fields)["label"]
    # valid_loss = onmt.utils.loss.build_loss_compute(model, tgt_field, opt, train=False)

    src_shards = split_corpus(opt.src, 0)

    for i, src_shard in enumerate(src_shards):
        logger.info("Evaluating shard %d." % i)
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
            readers=([src_reader, src_reader, src_reader]),
            data=([("sent1", sent1), ("sent2", sent2), ("label", label)]),
            dirs=[None, None, None],
            sort_key=inputters.str2sortkey[opt.data_type],
            filter_pred=None
        )

        data_iter = inputters.OrderedIterator(
            dataset=data,
            device=torch.device("cuda", opt.gpu) if opt.gpu > -1 else torch.device("cpu"),
            batch_size=opt.batch_size,
            train=False,
            sort=False,
            sort_within_batch=True,
            shuffle=False
        )

        with torch.no_grad():
            stats = onmt.utils.Statistics()

            for batch in data_iter:
                # src, src_lengths = batch.src if isinstance(batch.src, tuple) \
                #                    else (batch.src, None)
                # tgt = batch.tgt
                sent1, sent2 = batch.sent1, batch.sent2

                # F-prop through the model.
                outputs = model(sent1, sent2)

                output = model.classifier(outputs)

                # # Compute loss.
                # _, batch_stats = valid_loss(batch, outputs, None)
                #
                # # Update statistics.
                # stats.update(batch_stats)


        # translator.translate(
        #     src=src_shard,
        #     src_dir=opt.src_dir,
        #     sent1=sent1,
        #     sent2=sent2,
        #     label=label,
        #     batch_size=opt.batch_size,
        #     attn_debug=opt.attn_debug
        #     )


def _get_parser():
    parser = ArgumentParser(description='evaluate.py')

    opts.config_opts(parser)
    opts.evaluate_opts(parser)
    return parser


if __name__ == "__main__":
    parser = _get_parser()

    opt = parser.parse_args()
    main(opt)
