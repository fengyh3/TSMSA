# encoding=utf-8
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import crf

class CRF(object):
    def __init__(self, hidden_size, initializers,
                 num_labels, seq_length, scope):
        self.hidden_size = hidden_size
        self.initializers = initializers
        self.seq_length = seq_length
        self.num_labels = num_labels
        self.scope = scope

    def set_tensor(self, inputs, labels, lengths):
        self.inputs = inputs
        self.labels = labels
        self.lengths = lengths

    def add_crf_layer(self):
        with tf.variable_scope(self.scope + "_loss", reuse = tf.AUTO_REUSE):
            logits = self.project_layer(self.inputs)
            #crf
            loss, trans = self.crf_layer(logits)
            if self.scope == 'opinion':
                loss *= 2
            pred_ids, _ = crf.crf_decode(potentials=logits, transition_params=trans, sequence_length=self.lengths)
            return ((loss, logits, trans, pred_ids))

    def project_layer(self, inputs, name=None):
        with tf.variable_scope("project" if not name else name, reuse = tf.AUTO_REUSE):
            # project to score of tags
            with tf.variable_scope("logits", reuse = tf.AUTO_REUSE):
                output = tf.reshape(inputs, shape=[-1, self.hidden_size])

                W = tf.get_variable("W", shape=[self.hidden_size, self.num_labels],
                                    dtype=tf.float32, initializer=self.initializers)

                b = tf.get_variable("b", shape=[self.num_labels], dtype=tf.float32,
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
                initializer=self.initializers)
            log_likelihood, trans = tf.contrib.crf.crf_log_likelihood(
                inputs=logits,
                tag_indices=self.labels,
                transition_params=trans,
                sequence_lengths=self.lengths)
            return tf.reduce_mean(-log_likelihood), trans