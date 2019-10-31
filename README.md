This is testbed for deep learning experiments based on [OpenNMT](https://github.com/OpenNMT/OpenNMT-py) code

### preprocess
```
python preprocess.py \
    --train_src=data/semeval/train.tsv \
    --valid_src=data/semeval/dev.tsv \
    --save_data=data/semeval/temp
```

### word embedding
```
python embeddings_to_torch.py \
    -emb_file_both "glove_dir/glove.6B.100d.txt" \
    -dict_file "data/data.vocab.pt" \
    -output_file "data/embeddings"
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