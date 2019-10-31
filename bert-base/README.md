### BERT-base model
This code is based on [pytorch-transformers](https://github.com/huggingface/pytorch-transformers) from huggingface.

### run QNLI 
```
python run_classifier.py \
    --task_name=qnli \
    --data_dir=data/QNLI/ \
    --bert_model=bert-base-cased \
    --output_dir=qnli-hof \
    --do_train \
    --num_train_epoch=7 \
```

### run WikiQA
```
python run_classifier.py \
    --task_name=wikiqa \
    --data_dir=data/WikiQA/ \
    --bert_model=bert-base-cased \
    --output_dir=wikiqa-dwsx2 \
    --do_train \
    --num_train_epoch=10
```

### run SemEval2017
```
python run_classifier.py \
    --task_name=semeval \
    --data_dir=../data/semeval/ \
    --bert_model=bert-base-cased \
    --output_dir=semeval-base \
    --do_train \
    --num_train_epoch=10 \
```
