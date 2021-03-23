# encoding=utf-8
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import crf

class CRF(object):
    def __init__(self, hidden_size, droupout_rate, initializers,
                 num_labels, seq_length, scope, is_training):
        self.hidden_size = hidden_size
        self.droupout_rate = droupout_rate
        self.initializers = initializers
        self.seq_length = seq_length
        self.num_labels = num_labels
        self.scope = scope
        self.is_training = is_training

    def set_tensor(self, input_lstm, labels, lengths):
        self.input_lstm = tf.reshape(input_lstm, [1, self.seq_length, -1])
        self.labels = tf.reshape(labels, [1, self.seq_length])
        self.lengths = tf.reshape(lengths, [1])

    def add_crf_layer(self):
        with tf.variable_scope(self.scope, reuse = tf.AUTO_REUSE):
            logits = self.project_layer(self.input_lstm)
            #crf
            loss, trans = self.crf_layer(logits)
            if self.scope == 'opinion':
                loss *= 1
            # CRF decode, pred_ids 是一条最大概率的标注路径
            pred_ids, _ = crf.crf_decode(potentials=logits, transition_params=trans, sequence_length=self.lengths)
            return ((loss, logits, trans, pred_ids))

    def project_layer(self, lstm_outputs, name=None):
        with tf.variable_scope("project" if not name else name, reuse = tf.AUTO_REUSE):
            # project to score of tags
            with tf.variable_scope("logits", reuse = tf.AUTO_REUSE):
                output = tf.reshape(lstm_outputs, shape=[-1, self.hidden_size])

                W = tf.get_variable(self.scope + "W", shape=[self.hidden_size, self.num_labels],
                                    dtype=tf.float32, initializer=self.initializers.xavier_initializer())

                b = tf.get_variable(self.scope + "b", shape=[self.num_labels], dtype=tf.float32,
                                    initializer=tf.zeros_initializer())

                #pred = tf.nn.xw_plus_b(hidden, W, b)
                pred = tf.nn.xw_plus_b(output, W, b)
                pred = tf.reshape(pred, [-1, self.seq_length, self.num_labels])
            return pred

    def crf_layer(self, logits):
        with tf.variable_scope("crf_loss", reuse = tf.AUTO_REUSE):
            trans = tf.get_variable(
                "transitions",
                shape=[self.num_labels, self.num_labels],
                initializer=self.initializers.xavier_initializer())
            log_likelihood, trans = tf.contrib.crf.crf_log_likelihood(
                inputs=logits,
                tag_indices=self.labels,
                transition_params=trans,
                sequence_lengths=self.lengths)
            return tf.reduce_mean(-log_likelihood), trans