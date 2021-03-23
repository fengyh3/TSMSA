import tensorflow as tf
import math
import numpy as np
from crf import CRF
from utils import get_batches, get_predict_batches
from evaluate import evaluate

def get_shape_list(tensor):
    shape = tensor.shape.as_list()
    non_static_indexes = []
    for index, dim in enumerate(shape):
        if dim is None:
            non_static_indexes.append(index)

    if not non_static_indexes:
        return shape

    dyn_shape = tf.shape(tensor)
    for index in non_static_indexes:
        shape[index] = dyn_shape[index]
    return shape

def reshape_to_matrix(input_tensor):
    width = input_tensor.shape[-1]
    output_tensor = tf.reshape(input_tensor, [-1, width])
    return output_tensor

def reshape_from_matrix(output_tensor, orig_shape_list):
    output_shape = get_shape_list(output_tensor)
    orig_dims = orig_shape_list[0:-1]
    width = output_shape[-1]
    return tf.reshape(output_tensor, orig_dims + [width])

def create_initializer(initializer_range = 0.02):
    return tf.truncated_normal_initializer(stddev=initializer_range)

def gelu(x):
    cdf = 0.5 * (1.0 + tf.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3))))
    return x * cdf

def layer_norm(input_tensor, name = None):
    return tf.contrib.layers.layer_norm(inputs=input_tensor, begin_norm_axis = -1, begin_params_axis=-1, scope=name)

def layer_norm_and_dropout(input_tensor, dropout_prob, name = None):
    output_tensor = layer_norm(input_tensor, name)
    output_tensor = tf.nn.dropout(output_tensor, 1.0 - dropout_prob)
    return output_tensor

def create_attention_mask_from_input_mask(from_tensor, to_mask):
    from_shape = get_shape_list(from_tensor)
    batch_size = from_shape[0]
    from_seq_length = from_shape[1]

    to_shape = get_shape_list(to_mask)
    to_seq_length = to_shape[1]

    to_mask = tf.cast(tf.reshape(to_mask, [batch_size, 1, to_seq_length]), tf.float32)
    broadcast_ones = tf.ones(shape=[batch_size, from_seq_length, 1], dtype=tf.float32)
    mask = broadcast_ones * to_mask
    return mask

class BaseModel(object):
    def __init__(self, batch_size=32, max_seq_len=100, num_attention_heads=4, size_per_head=64, num_hidden_layers=1,
                intermediate_size=512, initializer_range=0.02, emb_path=None, num_labels=None):
        tf.set_random_seed(-1)
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.num_attention_heads = num_attention_heads
        self.size_per_head = size_per_head
        self.num_hidden_layers = num_hidden_layers
        self.intermediate_size = intermediate_size
        self.initializer_range = initializer_range
        self.num_labels = num_labels

        embedding_numpy = np.load(emb_path)
        self.embedding_table = tf.get_variable(name='embedding_table', shape=embedding_numpy.shape,
                                              initializer=tf.constant_initializer(embedding_numpy), trainable=False)
        #self.position_embeddings = tf.get_variable('position_embeddings_table', shape = [self.max_seq_len, embedding_numpy.shape[-1]], initializer=create_initializer(self.initializer_range))
        '''
        self.embedding_table = tf.get_variable(name='embedding_table', shape=[emb_path, 112],
                                              initializer=create_initializer(self.initializer_range), trainable=True)
        '''
        self.position_embeddings = tf.get_variable('position_embeddings_table', shape = [self.max_seq_len, 12], initializer=create_initializer(self.initializer_range))
        self.set_placeholder()

    def optimize(self, num_train_steps):
        with tf.variable_scope('optimize'):
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    def set_placeholder(self):
        self.input_ids = tf.placeholder(dtype = tf.int32, shape = [None, self.max_seq_len], name = 'input_ids')
        self.labels = tf.placeholder(dtype = tf.int32, shape = [None, self.max_seq_len], name = 'labels')
        self.hidden_dropout_placeholder = tf.placeholder(dtype = tf.float32, shape = [], name = 'hidden_dropout')
        self.attention_dropout_placeholder = tf.placeholder(dtype = tf.float32, shape = [], name = 'attention_dropout')

    def embedding_layer(self):
        batch_size = get_shape_list(self.input_ids)[0]
        with tf.variable_scope("word_embeddings"):
            embedding_output = tf.nn.embedding_lookup(self.embedding_table, ids=self.input_ids, name = 'word_embeddings')
            embedding_output = tf.layers.dense(embedding_output, 116, activation=gelu, kernel_initializer=create_initializer(self.initializer_range))

        with tf.variable_scope("position_embeddings"):
            position_embeddings_shape = get_shape_list(self.position_embeddings)
            tmp = tf.zeros([batch_size, position_embeddings_shape[0], position_embeddings_shape[1]])
            tmp += tf.reshape(self.position_embeddings, shape=[1, position_embeddings_shape[0], position_embeddings_shape[1]])
            embedding_output = tf.concat(axis = -1, values=[tmp, embedding_output])
            #embedding_output += tf.reshape(self.position_embeddings, shape=[1, position_embeddings_shape[0], position_embeddings_shape[1]])

        return embedding_output

    def attention_layer(self, from_tensor, to_tensor, input_shape, attention_mask=None,
                        num_attention_heads=1, size_per_head=512, query_act=None,
                        key_act=None, value_act=None, attention_probs_dropout_prob=0.0,
                        initializer_range=0.02, batch_size=None, from_seq_length=None,
                        to_seq_length=None, do_return_2d_tensor = False):
        '''
        from_tensor: [batch_size, from_seq_length, from_width]
        '''
        def transpose_for_scores(input_tensor, batch_size, num_attention_heads, seq_length, width):
            output_tensor = tf.reshape(input_tensor, [batch_size, seq_length, num_attention_heads, width])
            output_tensor = tf.transpose(output_tensor, [0, 2, 1, 3])
            return output_tensor


        #from_shape = get_shape_list(from_tensor)
        #to_shape = get_shape_list(to_tensor)

        batch_size = input_shape[0]
        from_seq_length = input_shape[1]
        to_seq_length = input_shape[1]

        #from_tensor_2d = reshape_to_matrix(from_tensor)
        #to_tensor_2d = reshape_to_matrix(to_tensor)
        from_tensor_2d = from_tensor
        to_tensor_2d = to_tensor

        # query_layer = [batch_size * seq_len, num_head * size_per_head]
        query_layer = tf.layers.dense(from_tensor_2d, num_attention_heads * size_per_head,
                                      activation = query_act, name = 'query', kernel_initializer = create_initializer(initializer_range))
        key_layer = tf.layers.dense(to_tensor_2d, num_attention_heads * size_per_head,
                                      activation = key_act, name = 'key', kernel_initializer = create_initializer(initializer_range))
        value_layer = tf.layers.dense(to_tensor_2d, num_attention_heads * size_per_head,
                                      activation = value_act, name = 'value', kernel_initializer = create_initializer(initializer_range))
        query_layer = transpose_for_scores(query_layer, batch_size, num_attention_heads, 
                                            from_seq_length, size_per_head)
        key_layer = transpose_for_scores(key_layer, batch_size, num_attention_heads,
                                        to_seq_length, size_per_head)
        #[0, 2, 1, 3] * [0, 2, 3, 1] = [0, 2, 1, 1] 0 1 2 3分别对应transpose_for_scores中的axis
        attention_scores = tf.matmul(query_layer, key_layer, transpose_b = True)
        attention_scores = tf.multiply(attention_scores, 1.0 / math.sqrt(float(size_per_head)))

        if attention_mask is not None:
            attention_mask = tf.expand_dims(attention_mask, axis = [1])
            adder = (1.0 - tf.cast(attention_mask, tf.float32)) * -10000.0
            attention_scores += adder

        attention_probs = tf.nn.softmax(attention_scores)
        if attention_probs_dropout_prob is not None and attention_probs_dropout_prob != 0.0:
            attention_probs = tf.nn.dropout(attention_probs, 1.0 - attention_probs_dropout_prob)

        value_layer = tf.reshape(value_layer, [batch_size, to_seq_length, num_attention_heads, size_per_head])
        value_layer = tf.transpose(value_layer, [0, 2, 1, 3])

        context_layer = tf.matmul(attention_probs, value_layer)
        context_layer = tf.transpose(context_layer, [0, 2, 1, 3])

        if do_return_2d_tensor:
            return tf.reshape(context_layer, [batch_size * from_seq_length, num_attention_heads * size_per_head])
        else:
            return tf.reshape(context_layer, [batch_size, from_seq_length, num_attention_heads * size_per_head])

    def transformer(self, input_tensor, attention_mask = None):
        hidden_size = self.num_attention_heads * self.size_per_head
        input_shape = get_shape_list(input_tensor)
        batch_size = input_shape[0]
        seq_length = input_shape[1]
        input_width = input_shape[2]

        prev_output = reshape_to_matrix(input_tensor)
        all_layer_outputs = []

        for layer_idx in range(self.num_hidden_layers):
            with tf.variable_scope("layer_%d" % layer_idx):
                layer_input = prev_output

                with tf.variable_scope("attention"):
                    attention_heads = []
                    with tf.variable_scope("self"):
                        attention_head = self.attention_layer(from_tensor = layer_input, to_tensor = layer_input, input_shape=input_shape, 
                                                              attention_mask = attention_mask, num_attention_heads=self.num_attention_heads,
                                                              size_per_head=self.size_per_head, attention_probs_dropout_prob=self.attention_dropout_placeholder,
                                                              initializer_range=self.initializer_range, do_return_2d_tensor=True, batch_size=batch_size,
                                                              from_seq_length=seq_length, to_seq_length=seq_length)
                        attention_heads.append(attention_head)

                    attention_output = None
                    if len(attention_heads) == 1:
                        attention_output = attention_heads[0]
                    else:
                        attention_output = tf.concat(attention_heads, axis = -1)

                    with tf.variable_scope("output"):
                        attention_output = tf.layers.dense(attention_output, hidden_size, activation=gelu, kernel_initializer = create_initializer(self.initializer_range))
                        attention_output = tf.nn.dropout(attention_output, 1.0 - self.hidden_dropout_placeholder)
                        attention_output = layer_norm(attention_output + layer_input)
                '''
                with tf.variable_scope("intermediate"):
                    intermediate_output = tf.layers.dense(attention_output, self.intermediate_size, activation=gelu, kernel_initializer=create_initializer(self.initializer_range))
                
                with tf.variable_scope("output"):
                    #layer_output = tf.layers.dense(intermediate_output, hidden_size, kernel_initializer=create_initializer(self.initializer_range))
                    layer_output = tf.layers.dense(attention_output, hidden_size, activation=gelu, kernel_initializer=create_initializer(self.initializer_range))
                    layer_output = tf.nn.dropout(layer_output, 1.0 - self.hidden_dropout_placeholder)
                    layer_output = layer_norm(layer_output + attention_output)
                    prev_output = layer_output
                    all_layer_outputs.append(layer_output)
                '''
                prev_output = attention_output
                all_layer_outputs.append(attention_output)
        final_output = reshape_from_matrix(prev_output, input_shape)
        return final_output

    def model(self):
        embedding_output = self.embedding_layer()
        used = tf.sign(tf.abs(self.input_ids))
        lengths = tf.reduce_sum(used, reduction_indices=1)
        attention_mask = create_attention_mask_from_input_mask(self.input_ids, used)
        final_output = self.transformer(embedding_output, attention_mask)

        crf_layer = CRF(self.num_attention_heads * self.size_per_head, create_initializer(self.initializer_range), self.num_labels, self.max_seq_len)
        crf_layer.set_tensor(final_output, self.labels, lengths)
        self.loss, self.logits, self.trans, self.pred_ids = crf_layer.add_crf_layer()

    def train(self, x, y, dev_x, dev_y, epochs, hidden_dropout=0.1, attention_dropout=0.1, learning_rate=0.02, label_list=None):
        flag = 0
        if len(x) % self.batch_size != 0:
            flag = 1
        num_train_steps = (len(x) / self.batch_size + flag) * epochs
        self.learning_rate = learning_rate
        self.model()
        self.optimize(num_train_steps)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        losses = []
        for epoch in range(epochs):
            batches_x, batches_y = get_batches(self.batch_size, x, y, is_random=True)
            loss_sum = 0.0
            for batch_x, batch_y in zip(batches_x, batches_y):
                feed_dict = {self.input_ids:batch_x, self.labels:batch_y, self.attention_dropout_placeholder:attention_dropout, self.hidden_dropout_placeholder:hidden_dropout}
                _, loss, pred_labels = self.sess.run([self.optimizer, self.loss, self.pred_ids], feed_dict = feed_dict)
                loss_sum += loss
            print('epoch %d loss: %s' % (epoch, loss_sum))
            losses.append(loss_sum)
            
            predict_ids = self.predict(dev_x)
            id2label = {}
            for idx, label in enumerate(label_list):
                id2label[idx + 1] = label

            labels = []
            for idx, label in enumerate(dev_y):
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
                    if predict[i] != 0:
                        pred.append(id2label[predict[i]])
                    else:
                        break
                        #pred.append('O')
                predict_labels.append(pred)

            evaluate(predict_labels, labels)
            

    def predict(self, x, y=None):
        batches_x = get_predict_batches(self.batch_size, x)
        results = []
        for batch_x in batches_x:
            feed_dict = {self.input_ids:batch_x, self.attention_dropout_placeholder:0.0, self.hidden_dropout_placeholder:0.0}
            pred_labels = self.sess.run([self.pred_ids], feed_dict=feed_dict)
            results.extend(pred_labels[0])
        return results

    def save(self, save_path):
        pass

    def load(self, file_path):
        pass