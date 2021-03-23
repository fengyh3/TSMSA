import tensorflow as tf
from base_model import BaseModel
from utils import read_data, data_process
from evaluate import evaluate
import numpy as np
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    np.random.seed(1)
    train_file = '../data_AOPE/20data/14lap/train.csv'
    test_file = '../data_AOPE/20data/14lap/test.csv'
    vocab_file = '../embedding/vocab.txt'
    embed_file = '../embedding/embedding.npy'
    dev_mode = True
    train_x_aspect, train_y_aspect, train_x_opinion, train_y_opinion = read_data(train_file)
    test_x_aspect, test_y_aspect, test_x_opinion, test_y_opinion = read_data(test_file)
    opinion_label_list = ['B', 'I', 'O', '[SEP]']
    aspect_label_list = ['B', 'I', 'O', '[SEP]']
    train_x_aspect, train_y_aspect = data_process(train_x_aspect, train_y_aspect, vocab_file, aspect_label_list, 100)
    train_x_opinion, train_y_opinion = data_process(train_x_opinion, train_y_opinion, vocab_file, opinion_label_list, 100)
    test_x_aspect, test_y_aspect = data_process(test_x_aspect, test_y_aspect, vocab_file, aspect_label_list, 100)
    test_x_opinion, test_y_opinion = data_process(test_x_opinion, test_y_opinion, vocab_file, opinion_label_list, 100)

    if dev_mode:
    #use dev set for tuning hyper-parameters.
        train_x_aspect, dev_x_aspect, train_y_aspect, dev_y_aspect = train_test_split(train_x_aspect, train_y_aspect, test_size=0.2)
        train_x_opinion, dev_x_opinion, train_y_opinion, dev_y_opinion = train_test_split(train_x_opinion, train_y_opinion, test_size=0.2)

    model = BaseModel(batch_size=64, max_seq_len=100, num_attention_heads=4, size_per_head=32, num_hidden_layers=6,
                intermediate_size=256, initializer_range=0.02, emb_path=embed_file, aspect_num_labels=len(aspect_label_list) + 1, opinion_num_labels=len(opinion_label_list) + 1)
    
    if dev_mode:
        model.train(train_x_aspect, train_y_aspect, train_x_opinion, train_y_opinion, dev_x_aspect, dev_y_aspect, dev_x_opinion, dev_y_opinion,
                    epochs=300, attention_dropout=0.5, hidden_dropout=0.5, learning_rate=0.001, aspect_label_list=aspect_label_list, opinion_label_list=opinion_label_list)
    else:
        model.train(train_x_aspect, train_y_aspect, train_x_opinion, train_y_opinion, test_x_aspect, test_y_aspect, test_x_opinion, test_y_opinion,
                    epochs=300, attention_dropout=0.5, hidden_dropout=0.5, learning_rate=0.001, aspect_label_list=aspect_label_list, opinion_label_list=opinion_label_list)
    '''
    predict_ids = model.predict(test_x)
    id2label = {}
    for idx, label in enumerate(label_list):
        id2label[idx + 1] = label

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
