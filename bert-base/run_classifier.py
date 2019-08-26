# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import random

import numpy as np
import math
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from torch.nn import CrossEntropyLoss, MSELoss, Sigmoid, KLDivLoss, Softmax, LogSoftmax
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score

from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.modeling import BertForSequenceClassification, BertConfig
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule

from data_processor import QnliProcessor, WikiQAProcessor, SemevalProcessor
from wikiqa_eval import wikiqa_eval
from semeval_eval import semeval_eval

logger = logging.getLogger(__name__)
nnSoftmax = Softmax(dim=0)
nnLogSoftmax = LogSoftmax(dim=0)


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, weight=0.01, logit0=0.0, logit1=0.0):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.weight = weight
        self.logit0 = logit0
        self.logit1 = logit1


def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer, output_mode, sep=False):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label : i for i, label in enumerate(label_list)}

    features = []
    features_by_label = [[] for _ in range(len(label_list))]  # [[label 0 data], [label 1 data] ... []]

    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if output_mode == "classification":
            label_id = label_map[example.label]
        elif output_mode == "regression":
            label_id = float(example.label)
        else:
            raise KeyError(output_mode)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id))

        features_by_label[label_id].append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id))

    if sep is False:
        return features
    else:
        assert len(features) == (len(features_by_label[0])+len(features_by_label[1]))
        logger.info(" total:  %d\tlabel 0: %d\tlabel 1: %d " % (
            len(features), len(features_by_label[0]), len(features_by_label[1])))
        return features, features_by_label


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }


def pearson_and_spearman(preds, labels):
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "corr": (pearson_corr + spearman_corr) / 2,
    }


def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if task_name == "cola":
        return {"mcc": matthews_corrcoef(labels, preds)}
    elif task_name == "sst-2":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "mrpc":
        return acc_and_f1(preds, labels)
    elif task_name == "sts-b":
        return pearson_and_spearman(preds, labels)
    elif task_name == "qqp":
        return acc_and_f1(preds, labels)
    elif task_name == "mnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "mnli-mm":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "qnli":
        return acc_and_f1(preds, labels)
    elif task_name == "rte":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "wnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "semeval":
        return acc_and_f1(preds, labels)
    else:
        raise KeyError(task_name)


def model_tokenizer_loader(args, num_labels, pre_trained=False):
    if args.model_name.split("-")[0] == 'bert':
        assert args.model_name in ["bert-base-uncased", "bert-large-uncased",
                                   "bert-base-cased", "bert-large-cased",
                                   "bert-base-multilingual-uncased",
                                   "bert-base-multilingual-cased", "bert-base-chinese."]
        if pre_trained is True:
            model = BertForSequenceClassification.from_pretrained(args.output_dir, num_labels=num_labels)
            tokenizer = BertTokenizer.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        else:
            cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed_{}'.format(args.local_rank))
            model = BertForSequenceClassification.from_pretrained(args.model_name, cache_dir=cache_dir, num_labels=num_labels)
            tokenizer = BertTokenizer.from_pretrained(args.model_name, do_lower_case=args.do_lower_case)

    return model, tokenizer


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--model_name", default=None, type=str, required=True)
    parser.add_argument("--bert_model", default=None, type=str, required=False,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--do_sampling', type=bool, default=False)
    parser.add_argument('--do_histloss', type=bool, default=False)
    # parser.add_argument('--sampling_size', type=int, default=5000)
    parser.add_argument('--major_spl_size', type=int, default=0, help="sampling size for major class")
    parser.add_argument('--minor_cls_size', type=int, default=0, help="size of minor class")
    # parser.add_argument('--pre_trained_model', type=str, default='', help='pre-trained model for eval')
    args = parser.parse_args()

    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    processors = {
        # "cola": ColaProcessor,
        # "mnli": MnliProcessor,
        # "mnli-mm": MnliMismatchedProcessor,
        # "mrpc": MrpcProcessor,
        # "sst-2": Sst2Processor,
        # "sts-b": StsbProcessor,
        # "qqp": QqpProcessor,
        "qnli": QnliProcessor,
        # "rte": RteProcessor,
        # "wnli": WnliProcessor,
        "wikiqa": WikiQAProcessor,
        "semeval": SemevalProcessor,
    }

    output_modes = {
        "cola": "classification",
        "mnli": "classification",
        "mrpc": "classification",
        "sst-2": "classification",
        "sts-b": "regression",
        "qqp": "classification",
        "qnli": "classification",
        "rte": "classification",
        "wnli": "classification",
        "wikiqa": "classification",
        "semeval": "classification",
    }

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')

    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    # if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
    #     raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()
    output_mode = output_modes[task_name]

    label_list = processor.get_labels()
    num_labels = len(label_list)

    if args.do_train:
        # Prepare model
        cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed_{}'.format(args.local_rank))
        # model = BertForSequenceClassification.from_pretrained(args.bert_model, cache_dir=cache_dir, num_labels=num_labels)
        model, tokenizer = model_tokenizer_loader(args, num_labels=num_labels)
        # tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

        if args.fp16:
            model.half()
        model.to(device)
        if args.local_rank != -1:
            try:
                from apex.parallel import DistributedDataParallel as DDP
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

            model = DDP(model)
        elif n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # Prepare data loader
        train_examples = processor.get_train_examples(args.data_dir)

        features_by_label = ""
        if args.do_sampling is True:  # for num_train_optimization step
            train_steps_per_ep = math.ceil((args.major_spl_size + args.minor_cls_size) / args.train_batch_size)  # ceiling
            train_features, features_by_label = convert_examples_to_features(
                train_examples, label_list, args.max_seq_length, tokenizer, output_mode, sep=True)
            num_train_examples = args.major_spl_size + args.minor_cls_size
        else:
            train_features = convert_examples_to_features(
                train_examples, label_list, args.max_seq_length, tokenizer, output_mode)
            train_data, all_label_ids = get_tensor_dataset(train_features, output_mode)
            num_train_examples = len(train_examples)
            if args.local_rank == -1:
                _train_sampler = RandomSampler(train_data)
            else:
                _train_sampler = DistributedSampler(train_data)
            train_dataloader = DataLoader(train_data, sampler=_train_sampler, batch_size=args.train_batch_size)
            train_steps_per_ep = len(train_dataloader)

        # Prepare data for devset
        dev_examples = processor.get_dev_examples(args.data_dir)
        dev_features = convert_examples_to_features(
            dev_examples, label_list, args.max_seq_length, tokenizer, output_mode)
        dev_data, all_dev_label_ids = get_tensor_dataset(dev_features, output_mode)

        num_train_optimization_steps = train_steps_per_ep // args.gradient_accumulation_steps * args.num_train_epochs
        if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

        # Prepare optimizer

        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]

        # loss weight for historical objective function
        loss_weight = torch.nn.Parameter(torch.tensor([0.5], dtype=torch.float).to(device))

        optimizer_grouped_parameters[0]['params'].append(loss_weight)

        if args.fp16:
            try:
                from apex.optimizers import FP16_Optimizer
                from apex.optimizers import FusedAdam
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

            optimizer = FusedAdam(optimizer_grouped_parameters,
                                  lr=args.learning_rate,
                                  bias_correction=False,
                                  max_grad_norm=1.0)
            if args.loss_scale == 0:
                optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
            else:
                optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)
            warmup_linear = WarmupLinearSchedule(warmup=args.warmup_proportion,
                                                 t_total=num_train_optimization_steps)

        else:
            optimizer = BertAdam(optimizer_grouped_parameters,
                                 lr=args.learning_rate,
                                 warmup=args.warmup_proportion,
                                 t_total=num_train_optimization_steps)

        global_step = 0
        nb_tr_steps = 0
        tr_loss = 0

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", num_train_examples)
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)

        model.train()
        for ep in range(1, int(args.num_train_epochs)+1):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0

            if args.do_sampling is True:
                logger.info("***** [epoch %d] (sampling) get new dataloader ... *****" % ep)
                train_dataloader = get_sampling_dataloader(args, features_by_label)

            if args.do_histloss is True and ep > 1:
                train_data, _ = get_tensor_dataset(train_features, output_mode)
                if args.local_rank == -1:
                    _train_sampler = RandomSampler(train_data)
                else:
                    _train_sampler = DistributedSampler(train_data)
                train_dataloader = DataLoader(train_data, sampler=_train_sampler, batch_size=args.train_batch_size)

            logger.info("***** [epoch %d] trainig iteration starts ... *****" % ep)
            for step, batch in enumerate(train_dataloader):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids, logit0, logit1 = batch

                # define a new function to compute loss values for both output_modes
                logits = model(input_ids, segment_ids, input_mask, labels=None)

                if output_mode == "classification":
                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))

                    if args.do_histloss is True and ep > 1:
                        # loss_weight = loss_weight.clamp(0, 1)
                        pre_logit = torch.cat([logit0.unsqueeze(1), logit1.unsqueeze(1)], dim=1)
                        loss_pre = loss_fct(pre_logit.view(-1, num_labels), label_ids.view(-1))
                        pre_dist = nnSoftmax(pre_logit)[:, 1]
                        cur_dist = nnLogSoftmax(logits)[:, 1]
                        loss_gap = KLDivLoss()(cur_dist, pre_dist)
                        loss_weight = Sigmoid()(3 * (1 - loss_pre))  # 3 is magic number..
                        loss = loss_weight.item()*loss + (1-loss_weight.item())*(loss_pre + 10 * loss_gap)  # 10 is magic number ...

                elif output_mode == "regression":
                    loss_fct = MSELoss()
                    loss = loss_fct(logits.view(-1), label_ids.view(-1))

                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        # modify learning rate with special warm up BERT uses
                        # if args.fp16 is False, BertAdam is used that handles this automatically
                        lr_this_step = args.learning_rate * warmup_linear.get_lr(global_step, args.warmup_proportion)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1
            # end of epoch
            ##########################################################################
            # update weight in sampling experiments
            if args.do_sampling is True or args.do_histloss is True:
                logger.info("***** [epoch %d] update logits ... *****" % ep)
                train_features, features_by_label = update_logit(train_features, model, device)
            if args.do_histloss is True:
                logger.info("***** [epoch %d] (HOF) loss weight %.4f %.4f *****" % (ep, loss_weight.item(), 1-loss_weight.item()))
            ##########################################################################
            # eval with dev set.
            dev_sampler = SequentialSampler(dev_data)
            if task_name == 'wikiqa':
                dev_dataloader = DataLoader(dev_data, sampler=dev_sampler, batch_size=1)
                score = wikiqa_eval(ep, device, dev_examples, dev_dataloader, model, logger)
                score = str(round(score, 4))
            elif task_name == 'semeval':
                dev_dataloader = DataLoader(dev_data, sampler=dev_sampler, batch_size=1)
                score = semeval_eval(ep, device, dev_examples, dev_dataloader, model, logger, _type="dev")
                score = str(round(score, 4))
            else:
                dev_dataloader = DataLoader(dev_data, sampler=dev_sampler, batch_size=args.eval_batch_size)

                model.eval()
                dev_loss = 0
                nb_dev_steps = 0
                preds = []

                logger.info("***** [epoch %d] devset evaluating ... *****" % ep)
                for batch in dev_dataloader:
                    # input_ids, input_mask, segment_ids, label_ids, _, _ = batch
                    input_ids = batch[0].to(device)
                    input_mask = batch[1].to(device)
                    segment_ids = batch[2].to(device)
                    label_ids = batch[3].to(device)

                    with torch.no_grad():
                        logits = model(input_ids, segment_ids, input_mask, labels=None)

                    # create eval loss and other metric required by the task
                    if output_mode == "classification":
                        loss_fct = CrossEntropyLoss()
                        tmp_eval_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
                    elif output_mode == "regression":
                        loss_fct = MSELoss()
                        tmp_eval_loss = loss_fct(logits.view(-1), label_ids.view(-1))

                    dev_loss += tmp_eval_loss.mean().item()
                    nb_dev_steps += 1
                    if len(preds) == 0:
                        preds.append(logits.detach().cpu().numpy())
                    else:
                        preds[0] = np.append(
                            preds[0], logits.detach().cpu().numpy(), axis=0)

                dev_loss = dev_loss / nb_dev_steps
                preds = preds[0]
                if output_mode == "classification":
                    preds = np.argmax(preds, axis=1)
                elif output_mode == "regression":
                    preds = np.squeeze(preds)
                result = compute_metrics(task_name, preds, all_dev_label_ids.numpy())
                loss = tr_loss/global_step if args.do_train else None

                result['dev_loss'] = dev_loss
                result['loss'] = loss
                logger.info("***** [epoch %d] devset eval results *****" % ep)
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))

                if task_name == "qnli":
                    score = str(round(result['f1'], 4))
                else:
                    score = str(round(result['acc'], 4))
            model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
            output_model_file = os.path.join(args.output_dir, 'pytorch_model_%d_%s.bin' % (ep, score))
            torch.save(model_to_save.state_dict(), output_model_file)

    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Save a trained model, configuration and tokenizer
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self

        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
        output_config_file = os.path.join(args.output_dir, CONFIG_NAME)

        torch.save(model_to_save.state_dict(), output_model_file)
        model_to_save.config.to_json_file(output_config_file)

        if args.model_name.split("-")[0] == 'bert':
            tokenizer.save_vocabulary(args.output_dir)

        # Load a trained model and vocabulary that you have fine-tuned
        # model = BertForSequenceClassification.from_pretrained(args.output_dir, num_labels=num_labels)
        # tokenizer = BertTokenizer.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
    # else:
    #     model = BertForSequenceClassification.from_pretrained(args.bert_model, num_labels=num_labels)
    # model.to(device)

    if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Load a trained model and vocabulary that you have fine-tuned
        # model = BertForSequenceClassification.from_pretrained(args.output_dir, num_labels=num_labels)
        model, tokenizer = model_tokenizer_loader(args, num_labels=num_labels, pre_trained=True)
        # tokenizer = BertTokenizer.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        model.to(device)

        test_examples = processor.get_test_examples(args.data_dir)
        test_features = convert_examples_to_features(
            test_examples, label_list, args.max_seq_length, tokenizer, output_mode)
        test_data, all_label_ids = get_tensor_dataset(test_features, output_mode)

        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(test_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)

        # Run prediction for full data
        eval_sampler = SequentialSampler(test_data)
        if task_name == 'wikiqa':
            eval_dataloader = DataLoader(test_data, sampler=eval_sampler, batch_size=1)
            _ = wikiqa_eval(0, device, test_examples, eval_dataloader, model, logger)
        elif task_name == 'semeval':
            eval_dataloader = DataLoader(test_data, sampler=eval_sampler, batch_size=1)
            _ = semeval_eval(0, device, test_examples, eval_dataloader, model, logger, _type="test")
        else:
            eval_dataloader = DataLoader(test_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

            model.eval()
            eval_loss = 0
            nb_eval_steps = 0
            preds = []

            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                # input_ids, input_mask, segment_ids, label_ids, _, _ = batch
                input_ids = batch[0].to(device)
                input_mask = batch[1].to(device)
                segment_ids = batch[2].to(device)
                label_ids = batch[3].to(device)

                with torch.no_grad():
                    logits = model(input_ids, segment_ids, input_mask, labels=None)

                # create eval loss and other metric required by the task
                if output_mode == "classification":
                    loss_fct = CrossEntropyLoss()
                    tmp_eval_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
                elif output_mode == "regression":
                    loss_fct = MSELoss()
                    tmp_eval_loss = loss_fct(logits.view(-1), label_ids.view(-1))

                eval_loss += tmp_eval_loss.mean().item()
                nb_eval_steps += 1
                if len(preds) == 0:
                    preds.append(logits.detach().cpu().numpy())
                else:
                    preds[0] = np.append(
                        preds[0], logits.detach().cpu().numpy(), axis=0)

            eval_loss = eval_loss / nb_eval_steps
            preds = preds[0]
            if output_mode == "classification":
                preds = np.argmax(preds, axis=1)
            elif output_mode == "regression":
                preds = np.squeeze(preds)
            result = compute_metrics(task_name, preds, all_label_ids.numpy())
            loss = tr_loss/global_step if args.do_train else None

            result['eval_loss'] = eval_loss
            # result['global_step'] = global_step
            result['loss'] = loss

            output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                    writer.write("%s = %s\n" % (key, str(result[key])))

        # hack for MNLI-MM
        if task_name == "mnli":
            task_name = "mnli-mm"
            processor = processors[task_name]()

            if os.path.exists(args.output_dir + '-MM') and os.listdir(args.output_dir + '-MM') and args.do_train:
                raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
            if not os.path.exists(args.output_dir + '-MM'):
                os.makedirs(args.output_dir + '-MM')

            eval_examples = processor.get_dev_examples(args.data_dir)
            eval_features = convert_examples_to_features(
                eval_examples, label_list, args.max_seq_length, tokenizer, output_mode)
            logger.info("***** Running evaluation *****")
            logger.info("  Num examples = %d", len(eval_examples))
            logger.info("  Batch size = %d", args.eval_batch_size)
            all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
            all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
            all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
            all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)

            eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
            # Run prediction for full data
            eval_sampler = SequentialSampler(eval_data)
            eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

            model.eval()
            eval_loss = 0
            nb_eval_steps = 0
            preds = []

            for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader, desc="Evaluating"):
                input_ids = input_ids.to(device)
                input_mask = input_mask.to(device)
                segment_ids = segment_ids.to(device)
                label_ids = label_ids.to(device)

                with torch.no_grad():
                    logits = model(input_ids, segment_ids, input_mask, labels=None)
            
                loss_fct = CrossEntropyLoss()
                tmp_eval_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
            
                eval_loss += tmp_eval_loss.mean().item()
                nb_eval_steps += 1
                if len(preds) == 0:
                    preds.append(logits.detach().cpu().numpy())
                else:
                    preds[0] = np.append(
                        preds[0], logits.detach().cpu().numpy(), axis=0)

            eval_loss = eval_loss / nb_eval_steps
            preds = preds[0]
            preds = np.argmax(preds, axis=1)
            result = compute_metrics(task_name, preds, all_label_ids.numpy())
            loss = tr_loss/global_step if args.do_train else None

            result['eval_loss'] = eval_loss
            result['global_step'] = global_step
            result['loss'] = loss

            output_eval_file = os.path.join(args.output_dir + '-MM', "eval_results.txt")
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                    writer.write("%s = %s\n" % (key, str(result[key])))


def update_logit(train_features, model, device):
    train_data, _ = get_tensor_dataset(train_features, "classification")
    loader = DataLoader(train_data, sampler=SequentialSampler(train_data), batch_size=1024)
    global_logit_idx = 0
    for batch in loader:
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids = batch[0:4]

        with torch.no_grad():
            logits = model(input_ids, segment_ids, input_mask, labels=None)

        batch_size = logits.size(0)
        for i in range(batch_size):
            train_features[global_logit_idx+i].logit0 = logits[i][0].item()
            train_features[global_logit_idx+i].logit1 = logits[i][1].item()
            if label_ids[i] == 0:
                train_features[global_logit_idx+i].weight = logits[i][1].item()
            else:  # label_ids[i] == 1
                train_features[global_logit_idx+i].weight = logits[i][0].item()
        global_logit_idx += batch_size

    assert global_logit_idx == len(train_data)

    features_by_label = [[], []]
    for f in train_features:
        if f.label_id == 0:
            features_by_label[0].append(f)
        else:
            features_by_label[1].append(f)
    return train_features, features_by_label


def softmax(x):
    return np.exp(x)/sum(np.exp(x))


def get_sampling_dataloader(args, features_by_label):
    weight0 = softmax([i.weight for i in features_by_label[0]])
    weight1 = softmax([i.weight for i in features_by_label[1]])
    if len(features_by_label[0]) > len(features_by_label[1]):
        label_0 = np.random.choice(features_by_label[0], args.major_spl_size, replace=False, p=weight0)
        label_1 = np.random.choice(features_by_label[1], args.minor_cls_size, replace=False, p=weight1)
    else:
        label_0 = np.random.choice(features_by_label[0], args.minor_cls_size, replace=False, p=weight0)
        label_1 = np.random.choice(features_by_label[1], args.major_spl_size, replace=False, p=weight1)
    total = np.concatenate((label_0, label_1))
    train_data, _ = get_tensor_dataset(total, "classification")
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)
    return train_dataloader


def get_tensor_dataset(features, output_mode):
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)
    all_logit0 = torch.tensor([f.logit0 for f in features], dtype=torch.float)
    all_logit1 = torch.tensor([f.logit1 for f in features], dtype=torch.float)
    train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_logit0, all_logit1)
    return train_data, all_label_ids


if __name__ == "__main__":
    main()
