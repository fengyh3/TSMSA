import tensorflow as tf
from base_model import BaseModel
from utils import read_data, data_process, build_vocab
from evaluate import evaluate
from sklearn.model_selection import train_test_split
import numpy as np

if __name__ == '__main__':
    train_file = '../data_TOWE/19data/16res/train.csv'
    test_file = '../data_TOWE/19data/16res/test.csv'
    vocab_file = '../embedding/vocab.txt'
    embed_file = '../embedding/embedding.npy'
    dev_mode = True
    train_x, train_y = read_data(train_file)
    test_x, test_y = read_data(test_file)
    #vocab = build_vocab(train_x, test_x, vocab_file)
    label_list = ['B', 'I', 'O', '[SEP]']
    train_x, train_y = data_process(train_x, train_y, vocab_file, label_list, 100)
    
    if dev_mode:
    #use for dev set firstly.
        train_x, dev_x, train_y, dev_y = train_test_split(train_x, train_y, test_size=0.2)

    test_x, test_y = data_process(test_x, test_y, vocab_file, label_list, 100)

    model = BaseModel(batch_size=32, max_seq_len=100, num_attention_heads=4, size_per_head=32, num_hidden_layers=6,
                intermediate_size=256, initializer_range=0.02, emb_path=embed_file, num_labels=len(label_list) + 1)
    
    if dev_mode:
        #search hyper-parameters.
        model.train(train_x, train_y, dev_x, dev_y, epochs=300, attention_dropout=0.5, hidden_dropout=0.5, learning_rate=0.001, label_list=label_list)
    else:
        #if hyper-parameters are searched and fixed, use all train set to train the model.
        model.train(train_x, train_y, test_x, test_y, epochs=300, attention_dropout=0.5, hidden_dropout=0.5, learning_rate=0.001, label_list=label_list)
    '''
    predict_ids = model.predict(test_x)
    id2label = {}
    for idx, label in enumerate(label_list):
        id2label[idx + 1] = labelp

    labels = []
    for idx, label in enumerate(test_y):
        lab = []
        for l in label:
            if int(l) == 0:
                break
            lab.append(id2label[int(l)])
        labels.append(lab)

    predict_labels = []
    for idx, predict in enumerate(predict_ids):
        pred = []
        for i in range(len(labels[idx])):
            pred.append(id2label[predict[i]])
        predict_labels.append(pred)

    evaluate(predict_labels, labels)
    '''
