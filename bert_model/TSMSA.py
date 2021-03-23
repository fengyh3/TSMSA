# -*- coding:utf-8 -*-
# author:yhfeng
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow import set_random_seed
set_random_seed(-1)

import collections
import os
from bert import modeling
from bert import optimization_crf
from bert import tokenization
import tensorflow as tf
from tensorflow.python.ops import math_ops
from tensorflow.contrib import crf
import utils.tf_metrics
import pickle
from utils.crf import CRF
from tensorflow.contrib.layers.python.layers import initializers
import numpy as np
from evaluate import evaluate
import copy

flags = tf.flags


FLAGS = flags.FLAGS

flags.DEFINE_string(
    "data_dir", None,
    "The input datadir.",
)

flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model."
)

flags.DEFINE_string(
    "task_name", "opinion_ner", "The name of the task to train."
)

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written."
)

## Other parameters
flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model)."
)

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text."
)

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization."
)

flags.DEFINE_bool(
    "do_train", False,
    "Whether to run training."
)
flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool("do_predict", False,"Whether to run the model in inference mode on the test set.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 1, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 3.0, "Total number of training epochs to perform.")



flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 3000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")
tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")
flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text, label=None, label_type=None):
        """Constructs a InputExample.

        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text = text
        self.label = label
        self.label_type = label_type


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_ids, label_type):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        #0 for aspect, 1 for opinion
        self.label_type = label_type
        #self.label_mask = label_mask

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_data(cls, input_file):
        """Reads a BIO data."""
        combine_list = ["'re", "'s", "n't", "'ll", "'d", "'ve", "'m", "'have"]
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = []
            words = []
            labels = []
            label_type = None
            for line in f:
                contends = line.strip()
                if contends == '0' or contends == '1':
                    label_type = int(contends)
                    continue
                word = line.strip().split(' ')[0]
                label = line.strip().split(' ')[-1]
                if contends.startswith("-DOCSTART-"):
                    words.append('')
                    continue
                # if len(contends) == 0 and words[-1] == '。':
                if len(contends) == 0:
                    l = ' '.join([label for label in labels if len(label) > 0])
                    w = ' '.join([word for word in words if len(word) > 0])
                    lines.append([label_type, l, w])
                    words = []
                    labels = []
                    continue
                if word in combine_list:
                    words[-1] = words[-1] + word
                else:
                    words.append(word)
                    labels.append(label)
            return lines

class OpinionProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "train.csv")), "train"
        )

    def get_dev_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "dev.csv")), "dev"
        )

    def get_test_examples(self,data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "test.csv")), "test")

    def get_aspect_labels(self):
        # prevent potential bug for chinese text mixed with english text
        #return ['B-OP', 'I-OP', 'O', 'X', '[CLS]', '[SEP]']
        return ['B-ASP', 'B-OP', 'I-ASP', 'I-OP', 'O', 'X', '[CLS]', '[SEP]']


    def get_opinion_labels(self):
        # prevent potential bug for chinese text mixed with english text
        #return ['B-OP', 'I-OP', 'O', 'X', '[CLS]', '[SEP]']
        return ['B', 'I', 'O', 'X', '[CLS]', '[SEP]']

    def _create_example(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text = tokenization.convert_to_unicode(line[-1])
            label = tokenization.convert_to_unicode(line[1])
            label_type = line[0]
            examples.append(InputExample(guid=guid, text=text, label=label, label_type=label_type))
            np.random.shuffle(examples)
        return examples



def write_tokens(tokens,mode):
    if mode=="test":
        path = os.path.join(FLAGS.output_dir, "token_"+mode+".txt")
        wf = open(path,'a')
        for token in tokens:
            if token!="**NULL**":
                wf.write(token+'\n')
        wf.close()

def convert_single_example(ex_index, example, aspect_label_map, opinion_label_map, max_seq_length, tokenizer,mode):
    textlist = example.text.split(' ')
    labellist = example.label.split(' ')
    label_type = example.label_type
    tokens = []
    labels = []
    # print(textlist)
    tokens.append('[CLS]')
    labels.append('[CLS]')
    for i, word in enumerate(textlist):
        token = tokenizer.tokenize(word)
        if len(token) == 0:
            token = tokenizer.tokenize(',')
        label_1 = labellist[i]
        tokens.extend(token)
        # print(label_1)
        for m in range(len(token)):
            if m == 0:
                labels.append(label_1)
            else:
                labels.append("X")
        # print(tokens, labels)
    # tokens = tokenizer.tokenize(example.text)
    
    if len(tokens) >= max_seq_length:
        tokens = tokens[0:(max_seq_length - 1)]
        labels = labels[0:(max_seq_length - 1)]

    tokens.append('[SEP]')
    labels.append('[SEP]')
    '''
    if len(tokens) > max_seq_length:
        tokens = tokens[0:(max_seq_length)]
        labels = labels[0:(max_seq_length)]
    '''
    ntokens = []
    segment_ids = []
    label_ids = []
    for i, token in enumerate(tokens):
        ntokens.append(token)
        segment_ids.append(0)
        if label_type == 0:
            label_ids.append(aspect_label_map[labels[i]])
        else:
            label_ids.append(opinion_label_map[labels[i]])

    input_ids = tokenizer.convert_tokens_to_ids(ntokens)
    input_mask = [1] * len(input_ids)

    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        label_ids.append(0)
        ntokens.append("**NULL**")

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(label_ids) == max_seq_length

    if ex_index < 5:
        tf.logging.info("*** Example ***")
        tf.logging.info("guid: %s" % (example.guid))
        tf.logging.info("tokens: %s" % " ".join(
            [tokenization.printable_text(x) for x in tokens]))
        tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        tf.logging.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))
        tf.logging.info("label_type: %s" % str(label_type))

    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_ids=label_ids,
        label_type=label_type,
    )
    write_tokens(ntokens,mode)
    return feature


def filed_based_convert_examples_to_features(
        examples, aspect_label_list, opinion_label_list, max_seq_length, tokenizer, output_file,mode=None
):
    aspect_label_map = {}
    opinion_label_map = {}
    for (i, label) in enumerate(aspect_label_list,1):
        aspect_label_map[label] = i
    for (i, label) in enumerate(opinion_label_list,1):
        opinion_label_map[label] = i
    with open(FLAGS.output_dir + '/aspect_label2id.pkl','wb') as w:
        pickle.dump(aspect_label_map,w)
    with open(FLAGS.output_dir + '/opinion_label2id.pkl','wb') as w:
        pickle.dump(opinion_label_map,w)

    writer = tf.python_io.TFRecordWriter(output_file)
    for (ex_index, example) in enumerate(examples):
        if ex_index % 5000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))
        feature = convert_single_example(ex_index, example, aspect_label_map, opinion_label_map, max_seq_length, tokenizer,mode)
        
        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["label_ids"] = create_int_feature(feature.label_ids)
        features["label_type"] = create_int_feature([feature.label_type])
        #features["label_mask"] = create_int_feature(feature.label_mask)
        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())


def file_based_input_fn_builder(input_file, seq_length, is_training, drop_remainder):
    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "label_type": tf.FixedLenFeature([], tf.int64),
        # "label_ids":tf.VarLenFeature(tf.int64),
        #"label_mask": tf.FixedLenFeature([seq_length], tf.int64),
    }

    def _decode_record(record, name_to_features):
        example = tf.parse_single_example(record, name_to_features)
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t
        return example

    def input_fn(params):
        batch_size = params["batch_size"]
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)
        d = d.apply(tf.contrib.data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder
        ))
        return d
    return input_fn


def create_model(bert_config, is_training, input_ids, input_mask,
                 segment_ids, labels, aspect_num_labels, opinion_num_labels, label_type, use_one_hot_embeddings):
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings
    )
    output_layer = model.get_sequence_output()
    CLS_output = model.get_pooled_output()
    hidden_size = output_layer.shape[-1].value
    used = tf.sign(tf.abs(input_ids))
    lengths = tf.reduce_sum(used, reduction_indices=1)
    attention_mask = modeling.create_attention_mask_from_input_mask(input_ids, used)

    if is_training:
        droupout_rate = None
    else:
        droupout_rate = None


    def aspect_operation(crf, inputs, labels, lengths, attention_mask):
        inputs = tf.reshape(inputs, [1, inputs.shape[0], -1])
        crf.set_tensor(inputs, labels, lengths)
        rst = crf.add_crf_layer()
        return rst

    def opinion_operation(crf, inputs, labels, lengths, attention_mask):
        inputs = tf.reshape(inputs, [1, inputs.shape[0], -1])
        crf.set_tensor(inputs, labels, lengths)
        rst = crf.add_crf_layer()
        return rst

    aspect_crf = CRF(hidden_size=hidden_size,droupout_rate=droupout_rate,
                     initializers=initializers, num_labels=aspect_num_labels,
                     seq_length=FLAGS.max_seq_length, scope='aspect', is_training=is_training)

    opinion_crf = CRF(hidden_size=hidden_size, droupout_rate=droupout_rate,
                      initializers=initializers, num_labels=opinion_num_labels,
                      seq_length=FLAGS.max_seq_length, scope='opinion', is_training=is_training)
    indices = np.array([0, ])
    task = tf.gather(labels, indices, axis=-1)
    if is_training:
        batch_size = task.shape[0].value
    else:
        batch_size = FLAGS.predict_batch_size
    loss = []
    logits = []
    pred_ids = []
    standard = tf.constant(1)
    for i in range(batch_size):
        rst = tf.cond(tf.equal(label_type[i], standard), lambda: opinion_operation(opinion_crf, output_layer[i], labels[i],
                      lengths[i], attention_mask[i]), lambda: aspect_operation(aspect_crf, output_layer[i], labels[i], lengths[i], attention_mask[i]))
        loss.append(rst[0])
        logits.append(rst[1])
        pred_ids.append(rst[3])
    loss = tf.stack(loss)
    loss = tf.reduce_mean(loss)
    logits = tf.concat(logits, axis = 0)
    pred_ids = tf.concat(pred_ids, axis = 0)
    
    return ((loss, logits, pred_ids))
        
def model_fn_builder(bert_config, aspect_num_labels, opinion_num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
    def model_fn(features, labels, mode, params):
        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]
        label_type = features["label_type"]
        #label_mask = features["label_mask"]
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        (total_loss, logits, pred_ids) = create_model(
            bert_config, is_training, input_ids, input_mask,segment_ids, label_ids,
            aspect_num_labels, opinion_num_labels, label_type, use_one_hot_embeddings)
        tvars = tf.trainable_variables()
        scaffold_fn = None
        if init_checkpoint:
            (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars,init_checkpoint)
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
            if use_tpu:
                def tpu_scaffold():
                    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                    return tf.train.Scaffold()
                scaffold_fn = tpu_scaffold
            else:
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
        tf.logging.info("**** Trainable Variables ****")

        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)
        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = optimization_crf.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                scaffold_fn=scaffold_fn)
            '''
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op)
            '''
        elif mode == tf.estimator.ModeKeys.EVAL:
            def metric_fn(label_ids, label_type, logits):
                weight = tf.sequence_mask(FLAGS.max_seq_length)
                main_labels = [1,2,3,4,5,6,7,8]

                precision = tf_metrics.precision(label_ids, pred_ids, num_labels, main_labels, weight)
                recall = tf_metrics.recall(label_ids, pred_ids, num_labels, main_labels, weight)
                f = tf_metrics.f1(label_ids, pred_ids, num_labels, main_labels, weight)
                return {
                    "eval_precision":precision,
                    "eval_recall": recall,
                    "eval_f1": f,
                }
            eval_metrics = (metric_fn, [label_ids, label_type, logits])
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metrics=eval_metrics,
                scaffold_fn=scaffold_fn)
            '''
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metrics=eval_metrics)
            '''
        else:
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode = mode,predictions=pred_ids,scaffold_fn=scaffold_fn
            )
            '''
            output_spec = tf.estimator.EstimatorSpec(mode=mode, predictions=pred_ids)
            '''
        return output_spec
    return model_fn


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    #tf.set_random_seed(-1)
    processors = {
        "opinion_ner": OpinionProcessor,
    }
    '''
    if not FLAGS.do_train and not FLAGS.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")
    '''

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    if FLAGS.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (FLAGS.max_seq_length, bert_config.max_position_embeddings))

    task_name = FLAGS.task_name.lower()
    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))
    processor = processors[task_name]()

    aspect_label_list = processor.get_aspect_labels()
    opinion_label_list = processor.get_opinion_labels()

    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)
    
    tpu_cluster_resolver = None
    if FLAGS.use_tpu and FLAGS.tpu_name:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2

    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=FLAGS.master,
        model_dir=FLAGS.output_dir,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=FLAGS.iterations_per_loop,
            num_shards=FLAGS.num_tpu_cores,
            per_host_input_for_training=is_per_host))
    '''
    tf.logging.info('load estimator...')
    config = tf.ConfigProto(
        allow_soft_placement=True, log_device_placement=True, gpu_options={"allow_growth":True, "per_process_gpu_memory_fraction":0.5})
    run_config = tf.estimator.RunConfig(session_config=config,
        model_dir=FLAGS.output_dir, save_checkpoints_steps=FLAGS.save_checkpoints_steps)
    '''
    train_examples = None
    num_train_steps = None
    num_warmup_steps = None

    if FLAGS.do_train:
        train_examples = processor.get_train_examples(FLAGS.data_dir)
        num_train_steps = int(
            len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
        num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

    model_fn = model_fn_builder(
        bert_config=bert_config,
        aspect_num_labels=len(aspect_label_list)+1,
        opinion_num_labels=len(opinion_label_list)+1,
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=FLAGS.use_tpu,
        use_one_hot_embeddings=FLAGS.use_tpu)
    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=FLAGS.train_batch_size,
        eval_batch_size=FLAGS.eval_batch_size,
        predict_batch_size=FLAGS.predict_batch_size)
    '''
    estimator = tf.estimator.Estimator(model_fn=model_fn, config=run_config,
        params={"batch_size":FLAGS.train_batch_size})
    '''
    if FLAGS.do_train:
        train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
        filed_based_convert_examples_to_features(
            train_examples, aspect_label_list, opinion_label_list, FLAGS.max_seq_length, tokenizer, train_file)
        tf.logging.info("***** Running training *****")
        tf.logging.info("  Num examples = %d", len(train_examples))
        tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
        tf.logging.info("  Num steps = %d", num_train_steps)
        train_input_fn = file_based_input_fn_builder(
            input_file=train_file,
            seq_length=FLAGS.max_seq_length,
            is_training=True,
            drop_remainder=True)
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
    if FLAGS.do_eval:
        eval_examples = processor.get_dev_examples(FLAGS.data_dir)
        eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")
        filed_based_convert_examples_to_features(
            eval_examples, aspect_label_list, opinion_label_list, FLAGS.max_seq_length, tokenizer, eval_file)

        tf.logging.info("***** Running evaluation *****")
        tf.logging.info("  Num examples = %d", len(eval_examples))
        tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)
        eval_steps = None
        if FLAGS.use_tpu:
            eval_steps = int(len(eval_examples) / FLAGS.eval_batch_size)
        eval_drop_remainder = True if FLAGS.use_tpu else False
        eval_input_fn = file_based_input_fn_builder(
            input_file=eval_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=eval_drop_remainder)
        result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)
        output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
        with open(output_eval_file, "w", encoding = 'utf-8') as writer:
            tf.logging.info("***** Eval results *****")
            for key in sorted(result.keys()):
                tf.logging.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))
    if FLAGS.do_predict:
        token_path = os.path.join(FLAGS.output_dir, "token_test.txt")
        with open(FLAGS.output_dir + '/aspect_label2id.pkl','rb') as rf:
            aspect_label2id = pickle.load(rf)
            aspect_id2label = {value:key for key,value in aspect_label2id.items()}
        with open(FLAGS.output_dir + '/opinion_label2id.pkl','rb') as rf:
            opinion_label2id = pickle.load(rf)
            opinion_id2label = {value:key for key,value in opinion_label2id.items()}
        if os.path.exists(token_path):
            os.remove(token_path)
        predict_examples = processor.get_test_examples(FLAGS.data_dir)
        labels = []
        texts = []
        label_types = []
        for line in predict_examples:
            label = line.label
            text = line.text
            label_type = line.label_type
            label = label.split(' ')
            text = text.split(' ')
            labels.append(label)
            texts.append(text)
            label_types.append(label_type)

        predict_file = os.path.join(FLAGS.output_dir, "predict.tf_record")
        filed_based_convert_examples_to_features(predict_examples, aspect_label_list, opinion_label_list,
                                                FLAGS.max_seq_length, tokenizer,
                                                predict_file,mode="test")
                            
        tf.logging.info("***** Running prediction*****")
        tf.logging.info("  Num examples = %d", len(predict_examples))
        tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)
        if FLAGS.use_tpu:
            # Warning: According to tpu_estimator.py Prediction on TPU is an
            # experimental feature and hence not supported here
            raise ValueError("Prediction in TPU not supported")
        predict_drop_remainder = True if FLAGS.use_tpu else False
        predict_input_fn = file_based_input_fn_builder(
            input_file=predict_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=predict_drop_remainder)

        result = estimator.predict(input_fn=predict_input_fn)

        predict_results = []
        for idx, prediction in enumerate(result):
            tmp = []
            for i in range(1, len(prediction)):
                if prediction[i] == 0:
                    break
                if prediction[i] == 4 and label_types[idx] == 1:
                    continue
                if prediction[i] == 6 and label_types[idx] == 0:
                    continue
                if label_types[idx] == 0:
                    tmp.append(aspect_id2label[prediction[i]])
                else:
                    tmp.append(opinion_id2label[prediction[i]])
            predict_results.append(tmp[:-1])
        evaluate(predict_results, labels, label_types, evaluate_type=0)

if __name__ == "__main__":
    flags.mark_flag_as_required("data_dir")
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("output_dir")
    tf.app.run()
