# -*- coding:utf-8 -*-
# author: yhfeng

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
from bert import modeling
from bert import optimization
from bert import tokenization
import tensorflow as tf
from sklearn.metrics import f1_score,precision_score,recall_score
from tensorflow.python.ops import math_ops
from tensorflow.contrib import crf
import pickle
from tensorflow.contrib.layers.python.layers import initializers
import numpy as np
import time
import json
from utils.crf import CRF
from util import read_data
import copy
import nltk

flags = tf.flags


FLAGS = flags.FLAGS

flags.DEFINE_string(
    "bert_config_file", "bert_base/bert_config.json",
    "The config json file corresponding to the pre-trained BERT model."
)

flags.DEFINE_string(
    "models_file", "output",
    "The trained model file corresponding to the new task."
)

flags.DEFINE_string(
    "init_checkpoint", "model.ckpt-1928",
    "Initial checkpoint."
)

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text."
)

flags.DEFINE_integer(
    "max_seq_length", 100,
    "The maximum total input sequence length after WordPiece tokenization."
)

flags.DEFINE_integer("predict_batch_size", 1, "Total batch size for predict.")


flags.DEFINE_string("vocab_file", 'bert_base/vocab.txt',
                    "The vocabulary file that the BERT model was trained on.")


def convert_single_text_to_tensor(text, tokenizer, max_seq_length):
    label_split = [False for i in range(max_seq_length)]
    tokens = []
    tokens.append('[CLS]')
    index = 0
    for i, word in enumerate(text):
        index += 1
        token = tokenizer.tokenize(word)
        tokens.extend(token)
        for idx in range(index + 1, index + len(token)):
            label_split[idx] = True
            index += 1
    
    if len(tokens) >= max_seq_length:
        tokens = tokens[0:(max_seq_length - 1)]

    tokens.append('[SEP]')
    '''
    if len(tokens) > max_seq_length:
        tokens = tokens[0:(max_seq_length)]
    '''
    ntokens = []
    segment_ids = []
    for i, token in enumerate(tokens):
        ntokens.append(token)
        segment_ids.append(0)

    input_ids = tokenizer.convert_tokens_to_ids(ntokens)
    input_mask = [1] * len(input_ids)

    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    return (input_ids, input_mask, segment_ids, label_split)

def find_opinion(pred, text):
    opinion_start = []
    opinion_end = []
    flag = False
    for idx in range(len(text)):
        if pred[idx] == 4:
            continue
        if pred[idx] == 1:
            if flag:
                opinion_end.append(idx)
            opinion_start.append(idx)
            flag = True
        elif flag and pred[idx] != 2:
            opinion_end.append(idx)
            flag = False
    if flag:
        opinion_end.append(len(text))

    return opinion_start, opinion_end

def find_aspect(pred, text):
    aspect_start = []
    aspect_end = []
    flag = False
    for idx in range(len(text)):
        if pred[idx] == 6:
            continue
        if pred[idx] == 1:
            if flag:
                aspect_end.append(idx)
            aspect_start.append(idx)
            flag = True
        elif flag and pred[idx] != 3:
            aspect_end.append(idx)
            flag = False
    if flag:
        aspect_end.append(len(text))
    return aspect_start, aspect_end

tokenizer = tokenization.FullTokenizer(vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)
bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
init_checkpoint = FLAGS.models_file + '/' + FLAGS.init_checkpoint
aspect_label_list = ['B-ASP', 'B-OP', 'I-ASP', 'I-OP', 'O', 'X', '[CLS]', '[SEP]']
opinion_label_list = ['B', 'I', 'O', 'X', '[CLS]', '[SEP]']

configsession = tf.ConfigProto()
sess = tf.Session(config=configsession)
input_ids = tf.placeholder (shape=[None, FLAGS.max_seq_length],dtype=tf.int32,name="input_ids")
input_masks = tf.placeholder (shape=[None, FLAGS.max_seq_length],dtype=tf.int32,name="input_mask")
segment_ids = tf.placeholder (shape=[None,FLAGS.max_seq_length],dtype=tf.int32,name="segment_ids")
label_ids = tf.placeholder(shape=[None, FLAGS.max_seq_length], dtype=tf.int32, name="label_ids")
label_types = tf.placeholder(shape=[], dtype=tf.int32, name="label_types")


with sess.as_default():
    model = modeling.BertModel(
        config=bert_config,
        is_training=False,
        input_ids=input_ids,
        input_mask=input_masks,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=False)
    output_layer = model.get_sequence_output()
    CLS_output = model.get_pooled_output()
    hidden_size = output_layer.shape[-1].value
    used = tf.sign(tf.abs(input_ids))
    lengths = tf.reduce_sum(used, reduction_indices=1)
    attention_mask = modeling.create_attention_mask_from_input_mask(input_ids, used)

    aspect_crf = CRF(hidden_size=hidden_size, droupout_rate=1.0, initializers=initializers, num_labels=len(aspect_label_list) + 1,
                         seq_length=FLAGS.max_seq_length, scope='aspect', is_training=False)

    opinion_crf = CRF(hidden_size=hidden_size, droupout_rate=1.0, initializers=initializers, num_labels=len(opinion_label_list) + 1,
                          seq_length=FLAGS.max_seq_length, scope='opinion', is_training=False)


    def aspect_operation(crf, inputs, labels, lengths, attention_mask):
        #inputs = tf.reshape(inputs, [1, inputs.shape[0], -1])
        crf.set_tensor(inputs, labels, lengths)
        rst = crf.add_crf_layer()
        return rst

    def opinion_operation(crf, inputs, labels, lengths, attention_mask):
        #inputs = tf.reshape(inputs, [1, inputs.shape[0], -1])
        crf.set_tensor(inputs, labels, lengths)
        rst = crf.add_crf_layer()
        return rst


    pred_ids = []
    standard = tf.constant(1)
    rst = tf.cond(tf.equal(label_types, standard), lambda: opinion_operation(opinion_crf, output_layer, label_ids,
                    lengths, attention_mask), lambda: aspect_operation(aspect_crf, output_layer, label_ids, lengths, attention_mask))
    pred_ids.append(rst[-1])
    pred_ids = tf.concat(pred_ids, axis = 0)

    saver = tf.train.Saver()
    saver.restore(sess, init_checkpoint)

def opinion_mining(texts, tokenizer, data_file):
    results = []
    for text in texts:
        opinion = []
        raw_text = copy.deepcopy(text)
        input_id, input_mask, segment_id, label_split = convert_single_text_to_tensor(text, tokenizer, FLAGS.max_seq_length)
        input_id = [input_id]
        input_mask = [input_mask]
        segment_id = [segment_id]
        label_type = 0
        pred = sess.run(pred_ids, feed_dict={input_ids:input_id, input_masks:input_mask, segment_ids:segment_id, label_types:label_type})
        pred = pred[0]
        tmp = []
        for i in range(1, len(pred)):
            if pred[i] == 0:
                break
            if pred[i] == 6 and label_split[i]:
                continue
            tmp.append(pred[i])
        pred = tmp[:-1]
        #print(text)
        #print(pred)
        pred_task0 = copy.deepcopy(pred)
        aspect_start, aspect_end = find_aspect(pred_task0, text)

        label_type = 1
        for start, end in zip(aspect_start, aspect_end):
            new_text = text[:start] + ['‖'] + text[start:end] + ['‖'] + text[end:]
            input_id, input_mask, segment_id, label_split = convert_single_text_to_tensor(new_text, tokenizer, FLAGS.max_seq_length)
            input_id = [input_id]
            input_mask = [input_mask]
            segment_id = [segment_id]
            pred = sess.run(pred_ids, feed_dict={input_ids:input_id, input_masks:input_mask, segment_ids:segment_id, label_types:label_type})
            pred = pred[0]
            tmp = []
            for i in range(1, len(pred)):
                if pred[i] == 0:
                    break
                if pred[i] == 4 and label_split[i]:
                    continue
                tmp.append(pred[i])
            pred = tmp[:-1]
            opinion_start, opinion_end = find_opinion(pred, new_text)
            for s, e in zip(opinion_start, opinion_end):
                opinion.append(' '.join(text[start:end]) + ' ' + ' '.join(new_text[s : e]))
            
        results.append(opinion)

    return results


aspect_label_map = {}
for idx, l in enumerate(aspect_label_list):
    aspect_label_map[l] = idx + 1

opinion_label_map = {}
for idx, l in enumerate(opinion_label_list):
    opinion_label_map[l] = idx + 1

texts = []
data_file = './20data/14lap/test.csv'
group = {}
doc1, label1, doc2, label2 = read_data(data_file)
for idx, doc in enumerate(doc2):
    s = []
    for d in doc:
        if d != '‖':
            s.append(d)

    group[idx] = s
    if s not in texts:
        texts.append(s)

golden_pairs_map = {}
for idx, label in enumerate(label2):
    labs = []
    for l in label:
        labs.append(opinion_label_map[l])
    opinion_start, opinion_end = find_opinion(labs, doc2[idx])

    target_flag = False
    target = []
    for char in doc2[idx]:
        if char == '‖':
            if target_flag:
                break
            else:
                target_flag = True
                continue
        if target_flag:
            target.append(char)
    text = ' '.join(group[idx])
    for s, e in zip(opinion_start, opinion_end):
        tokens = []
        for i in range(s, e):
            tokens.append(doc2[idx][i])
        if golden_pairs_map.get(text) is None:
            golden_pairs_map[text] = [' '.join(target) + ' ' + ' '.join(tokens), ]
        else:
            golden_pairs_map[text].append(' '.join(target) + ' ' + ' '.join(tokens))
    if len(opinion_start) == 0:
        golden_pairs_map[text] = []

for idx, doc in enumerate(doc1):
    s = [t for t in doc]
    if s not in texts:
        texts.append(s)
        golden_pairs_map[' '.join(s)] = []


predict_pairs = opinion_mining(texts, tokenizer, data_file)

predict_counter = 0
golden_counter = 0
correct_counter = 0

for idx, text in enumerate(texts):
    text = ' '.join(text)
    golden_pair = golden_pairs_map[text]
    golden_counter += len(golden_pair)
    predict_pair = predict_pairs[idx]
    print(golden_pair)
    print(predict_pair)

    predict_counter += len(predict_pair)
    golden_map = {}
    for gp in golden_pair:
        if golden_map.get(gp) is None:
            golden_map[gp] = 1
        else:
            golden_map[gp] += 1
    
    for pp in predict_pair:
        if golden_map.get(pp) is not None:
            if golden_map[pp] > 0:
                correct_counter += 1
                golden_map[pp] -= 1

pair_precision = correct_counter / predict_counter
pair_recall = correct_counter / golden_counter
pair_f1 = (2 * pair_precision * pair_recall) / (pair_precision + pair_recall + 1e-10)
print('pair token-level: \nprecision: %s, recall: %s, f1: %s' % (pair_precision, pair_recall, pair_f1))
