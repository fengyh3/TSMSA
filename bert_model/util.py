import numpy as np

def read_data(path):
    with open(path, 'r', encoding = 'utf-8') as f:
        data = f.readlines()
    data = [d.strip() for d in data]

    combine_list = ["'re", "'s", "n't", "'ll", "'d", "'ve", "'m", "'have"]
    aspect_counter = 0
    opinion_counter = 0
    withA_counter = 0
    noA_counter = 0
    #combine_list = []
    label_types = []
    docs1 = []
    labels1 = []
    docs2 = []
    labels2 = []
    doc = []
    label = []
    flag = 0
    for line in data:
        if line == '1' or line == '0':
            flag = int(line)
            continue
        if line == '':
            if flag == 0:
                docs1.append(doc)
                labels1.append(label)
            else:
                docs2.append(doc)
                labels2.append(label)

            counter = 0
            FLAG = False
            if flag == 1:
                for l in label:
                    if l == 'B':
                        counter += 1
                    if l == '[SEP]':
                        FLAG = True
                if FLAG:
                    withA_counter += counter
                else:
                    noA_counter += counter

            doc = []
            label = []
            flag = 0
            continue
        word = line.split(' ')[0].lower()
        if word in combine_list:
            doc[-1] = doc[-1] + word
        else:
            doc.append(word)
            label.append(line.split(' ')[1])
        if line.split(' ')[1] == 'B':
            if flag == 0:
                aspect_counter += 1
            else:
                opinion_counter += 1
    return docs1, labels1, docs2, labels2

def convert_single_text_to_tensor(text, labels, label_lists, tokenizer, max_seq_length):
    label_map = {}
    for i in range(len(label_lists)):
        label_map[label_lists[i]] = i + 1
    tokens = []
    label_ids = [label_map['[CLS]'], ]
    tokens.append('[CLS]')
    for i, word in enumerate(text):
        token = tokenizer.tokenize(word)
        label_1 = label_map[labels[i]]
        tokens.extend(token)
        # print(label_1)
        for m in range(len(token)):
            if m == 0:
                label_ids.append(label_1)
            else:
                label_ids.append(label_map["X"])

    if len(tokens) >= max_seq_length:
        tokens = tokens[0:(max_seq_length - 1)]
        label_ids = label_ids[0:(max_seq_length - 1)]

    tokens.append('[SEP]')
    label_ids.append(label_map['[SEP]'])

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
        label_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(label_ids) == max_seq_length
    return (input_ids, input_mask, segment_ids, label_ids)

def get_batches(batch_size, input_ids, input_masks, segment_ids, y, is_random=True):
    if is_random:
        input_ids = np.array(input_ids)
        input_masks = np.array(input_masks)
        segment_ids = np.array(segment_ids)
        y = np.array(y)
        shuffle_idx = np.random.permutation(np.arange(len(input_ids)))
        input_ids = input_ids[shuffle_idx]
        input_masks = input_masks[shuffle_idx]
        segment_ids = segment_ids[shuffle_idx]
        y = y[shuffle_idx]

    batches_input_ids = []
    batches_input_masks = []
    batches_segment_ids = []
    batches_y = []
    for idx in range(int(len(input_ids) / batch_size)):
        batch_input_ids = input_ids[idx * batch_size : (idx + 1) * batch_size]
        batch_input_masks = input_masks[idx * batch_size : (idx + 1) * batch_size]
        batch_segment_ids = segment_ids[idx * batch_size : (idx + 1) * batch_size]
        batch_y = y[idx * batch_size : (idx + 1) * batch_size]
        batches_input_ids.append(batch_input_ids)
        batches_input_masks.append(batch_input_masks)
        batches_segment_ids.append(batch_segment_ids)
        batches_y.append(batch_y)

    if len(input_ids) % batch_size != 0:
        batches_input_ids.append(input_ids[-(len(input_ids) % batch_size) : ])
        batches_input_masks.append(input_masks[-(len(input_masks) % batch_size) : ])
        batches_segment_ids.append(segment_ids[-(len(segment_ids) % batch_size) : ])
        batches_y.append(y[-(len(y) % batch_size) : ])
    
    return batches_input_ids, batches_input_masks, batches_segment_ids, batches_y

def get_predict_batches(batch_size, input_ids, input_masks, segment_ids):
    batches_input_ids = []
    batches_input_masks = []
    batches_segment_ids = []

    for idx in range(int(len(input_ids) / batch_size)):
        batch_input_ids = input_ids[idx * batch_size : (idx + 1) * batch_size]
        batch_input_masks = input_masks[idx * batch_size : (idx + 1) * batch_size]
        batch_segment_ids = segment_ids[idx * batch_size : (idx + 1) * batch_size]
        batches_input_ids.append(batch_input_ids)
        batches_input_masks.append(batch_input_masks)
        batches_segment_ids.append(batch_segment_ids)

    if len(input_ids) % batch_size != 0:
        batches_input_ids.append(input_ids[-(len(input_ids) % batch_size) : ])
        batches_input_masks.append(input_masks[-(len(input_masks) % batch_size) : ])
        batches_segment_ids.append(segment_ids[-(len(segment_ids) % batch_size) : ])
    
    return batches_input_ids, batches_input_masks, batches_segment_ids