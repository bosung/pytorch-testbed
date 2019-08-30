This is testbed for deep learning experiments based on [OpenNMT](https://github.com/OpenNMT/OpenNMT-py) code

### preprocess
```
python preprocess.py \
    --train_src=data/semeval/train.tsv \
    --valid_src=data/semeval/dev.tsv \
    --save_data=data/semeval/temp
```

### train
```
python train.py \
    -data data/semeval/temp \
    -save_model temp-model \
    -gpu_ranks 0 \
    -early_stopping 5 \
    -pre_word_vecs_enc data/semeval/embeddings.enc.pt
```

### evaluate
```
python evaluate.py \
    -model temp-model_step_50000.pt \
    -src data/semeval/test.tsv \
    -gpu 0
```