# TSMSA
NAACL 2021 Accepted

title: Target-specified Sequence Labeling with Multi-head Self-attention for Target-oriented OpinionWords Extraction

## environment
tensorflow 1.13

Other package can use pip to install.

## Tips for base model:
1. hyper-parameter setting is in main.py.
2. using main.py to run the model.
3. the details of model are in base_model.py

## Tips for bert model:
1. pre-trained bert can be stored in bert_base file.
2. the model you trained can be stored in output file.
3. command of running (just like bert's):
```
python MTTSMSA.py --do_train=true --do_predict=true --data_dir=./AOPE_data/19data/14lap --vocab_file=./bert_base/vocab.txt --bert_config_file=./bert_base/bert_config.json --init_checkpoint=./bert_base/bert_model.ckpt --max_seq_length=100 --train_batch_size=32 --learning_rate=5e-5 --num_train_epochs=3.0 --output_dir=./output
```
