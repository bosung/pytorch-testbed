Pytorch testbed for deep learning experiments based on [OpenNMT](https://github.com/OpenNMT/OpenNMT-py) and [pytorch-transformers](https://github.com/huggingface/pytorch-transformers) from huggingface.

### ONMT 
- preprocess
```
python preprocess.py \
    --train_src=data/semeval/train.tsv \
    --valid_src=data/semeval/dev.tsv \
    --save_data=data/semeval/temp
```

- word embedding
```
python embeddings_to_torch.py \
    -emb_file_both "glove_dir/glove.6B.100d.txt" \
    -dict_file "data/data.vocab.pt" \
    -output_file "data/embeddings"
```

- train
```
python train.py \
    -data data/semeval/temp \
    -save_model temp-model \
    -gpu_ranks 0 \
    -early_stopping 5 \
    -pre_word_vecs_enc data/semeval/embeddings.enc.pt
```

- evaluate
```
python evaluate.py \
    -model temp-model_step_50000.pt \
    -src data/semeval/test.tsv \
    -gpu 0
```

### BERT

- run QNLI 
```
python run_classifier.py \
    --task_name=qnli \
    --data_dir=data/QNLI/ \
    --bert_model=bert-base-cased \
    --output_dir=qnli-hof \
    --do_train \
    --num_train_epoch=7 \
```

- run WikiQA
```
python run_classifier.py \
    --task_name=wikiqa \
    --data_dir=data/WikiQA/ \
    --bert_model=bert-base-cased \
    --output_dir=wikiqa-dwsx2 \
    --do_train \
    --num_train_epoch=10
```

- run SemEval2017
```
python run_classifier.py \
    --task_name=semeval \
    --data_dir=../data/semeval/ \
    --bert_model=bert-base-cased \
    --output_dir=semeval-base \
    --do_train \
    --num_train_epoch=10 \
```
