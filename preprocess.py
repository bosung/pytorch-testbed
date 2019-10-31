#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Pre-process Data / features files and build vocabulary
"""
import codecs
import glob
import sys
import gc
import torch
from functools import partial
from collections import Counter, defaultdict

from onmt.utils.logging import init_logger, logger
from onmt.utils.misc import split_corpus
import onmt.inputters as inputters
import onmt.opts as opts
from onmt.utils.parse import ArgumentParser
from onmt.inputters.inputter import _build_fields_vocab,\
                                    _load_vocab


def check_existing_pt_files(opt):
    """ Check if there are existing .pt files to avoid overwriting them """
    pattern = opt.save_data + '.{}*.pt'
    for t in ['train', 'valid']:
        path = pattern.format(t)
        if glob.glob(path):
            sys.stderr.write("Please backup existing pt files: %s, "
                             "to avoid overwriting them!\n" % path)
            sys.exit(1)


def build_save_dataset(corpus_type, fields, src_reader, opt):
    assert corpus_type in ['train', 'valid']

    if corpus_type == 'train':
        counters = defaultdict(Counter)
        srcs = opt.train_src
        ids = opt.train_ids
    else:
        srcs = [opt.valid_src]
        ids = [None]

    for src, maybe_id in zip(srcs, ids):
        logger.info("Reading source files: %s." % src)

        # src_shards = split_corpus(src, opt.shard_size)
        src_shards = split_corpus(src, 0)

        dataset_paths = []
        # if (corpus_type == "train" or opt.filter_valid):
        #     filter_pred = partial(
        #         inputters.filter_example, use_src_len=opt.data_type == "text",
        #         max_src_len=opt.src_seq_length, max_tgt_len=opt.tgt_seq_length)
        # else:
        filter_pred = None

        if corpus_type == "train":
            existing_fields = None
            if opt.src_vocab != "":
                try:
                    logger.info("Using existing vocabulary...")
                    existing_fields = torch.load(opt.src_vocab)
                except torch.serialization.pickle.UnpicklingError:
                    logger.info("Building vocab from text file...")
                    src_vocab, src_vocab_size = _load_vocab(
                        opt.src_vocab, "src", counters,
                        opt.src_words_min_frequency)
            else:
                src_vocab = None

        for i, _src_shard in enumerate(src_shards):  # not considered shard
            logger.info("Building shard %d." % i)
            src_shard = []
            for j, line in enumerate(_src_shard):
                if len(line.strip().split("\t")) == 6:
                    src_shard.append(line)
            _id = [line.strip().split("\t")[0] for line in src_shard[1:]]
            # _id = [i for i, line in enumerate(src_shard[1:], 1)]
            sent1 = [line.strip().split("\t")[3] for line in src_shard[1:]]
            sent2 = [line.strip().split("\t")[4] for line in src_shard[1:]]
            prelogit1 = [0.0 for _ in src_shard[1:]]
            prelogit2 = [0.0 for _ in src_shard[1:]]
            label = []
            for line in src_shard[1:]:
                token = line.strip().split("\t")[5]
                if token in ["Good", "entailment", "1", 1]:
                    label.append(1)
                else:
                    label.append(0)
            dataset = inputters.Dataset(
                fields,
                readers=([src_reader, src_reader, src_reader, src_reader, src_reader, src_reader]),
                data=([("id", _id), ("sent1", sent1), ("sent2", sent2),
                       ("label", label), ("prelogit1", prelogit1), ("prelogit2", prelogit2)]),
                # data=([("src", src_shard), ("tgt", tgt_shard)]
                #       if tgt_reader else [("src", src_shard)]),
                dirs=([None, None, None, None, None, None]),
                sort_key=inputters.str2sortkey[opt.data_type],
                filter_pred=filter_pred
            )
            if corpus_type == "train" and existing_fields is None:
                for ex in dataset.examples:
                    for name, field in fields.items():
                        if name in ["label", "id", "prelogit1", "prelogit2"]:
                            continue
                        try:
                            f_iter = iter(field)
                        except TypeError:
                            f_iter = [(name, field)]
                            all_data = [getattr(ex, name, None)]
                        else:
                            all_data = getattr(ex, name)
                        for (sub_n, sub_f), fd in zip(f_iter, all_data):
                            has_vocab = (sub_n == 'src' and src_vocab)
                            if (hasattr(sub_f, 'sequential')
                                    and sub_f.sequential and not has_vocab):
                                val = fd
                                counters[sub_n].update(val)
            if maybe_id:
                shard_base = corpus_type + "_" + maybe_id
            else:
                shard_base = corpus_type
            data_path = "{:s}.{:s}.{:d}.pt".\
                format(opt.save_data, shard_base, i)
            dataset_paths.append(data_path)

            logger.info(" * saving %sth %s data shard to %s."
                        % (i, shard_base, data_path))

            dataset.save(data_path)

            del dataset.examples
            gc.collect()
            del dataset
            gc.collect()

    if corpus_type == "train":
        vocab_path = opt.save_data + '.vocab.pt'
        if existing_fields is None:
            fields = _build_fields_vocab(
                fields, counters, opt.data_type,
                opt.share_vocab, opt.vocab_size_multiple,
                opt.src_vocab_size, opt.src_words_min_frequency)
        else:
            fields = existing_fields
        torch.save(fields, vocab_path)


def build_save_vocab(train_dataset, fields, opt):
    fields = inputters.build_vocab(
        train_dataset, fields, opt.data_type, opt.share_vocab,
        opt.src_vocab, opt.src_vocab_size, opt.src_words_min_frequency,
        opt.tgt_vocab, opt.tgt_vocab_size, opt.tgt_words_min_frequency,
        vocab_size_multiple=opt.vocab_size_multiple
    )
    vocab_path = opt.save_data + '.vocab.pt'
    torch.save(fields, vocab_path)


def count_features(path):
    """
    path: location of a corpus file with whitespace-delimited tokens and
                    ￨-delimited features within the token
    returns: the number of features in the dataset
    """
    with codecs.open(path, "r", "utf-8") as f:
        first_tok = f.readline().split(None, 1)[0]
        return len(first_tok.split(u"￨")) - 1


def main(opt):
    ArgumentParser.validate_preprocess_args(opt)
    torch.manual_seed(opt.seed)
    if not(opt.overwrite):
        check_existing_pt_files(opt)

    init_logger(opt.log_file)
    logger.info("Extracting features...")

    src_nfeats = 0
    # tgt_nfeats = 0
    for src in opt.train_src:
        src_nfeats += count_features(src) if opt.data_type == 'text' \
            else 0
    logger.info(" * number of source features: %d." % src_nfeats)

    logger.info("Building `Fields` object...")
    fields = inputters.get_fields(
        opt.data_type,
        src_nfeats,
        dynamic_dict=opt.dynamic_dict,
        src_truncate=opt.src_seq_length_trunc)

    src_reader = inputters.str2reader[opt.data_type].from_opt(opt)

    logger.info("Building & saving training data...")
    build_save_dataset('train', fields, src_reader, opt)

    if opt.valid_src:
        logger.info("Building & saving validation data...")
        build_save_dataset('valid', fields, src_reader, opt)


def _get_parser():
    parser = ArgumentParser(description='preprocess.py')

    opts.config_opts(parser)
    opts.preprocess_opts(parser)
    return parser


if __name__ == "__main__":
    parser = _get_parser()

    opt = parser.parse_args()
    main(opt)
