import numpy as np

def read_data(path):
    with open(path, 'r', encoding = 'utf-8') as f:
        data = f.readlines()
    data = [d.strip() for d in data]

    docs_aspect = []
    labels_aspect = []
    docs_opinion = []
    labels_opinion = []
    doc = []
    label = []
    flag = 0
    for line in data:
        if line == '1':
            flag = 1
            continue
        elif line == '0':
            flag = 0
            continue
        if line == '':
            if flag == 0:
                docs_aspect.append(doc)
                labels_aspect.append(label)
            elif flag == 1:
                docs_opinion.append(doc)
                labels_opinion.append(label)
            doc = []
            label = []
            continue
        doc.append(line.split(' ')[0])
        label.append(line.split(' ')[1])

    return docs_aspect, labels_aspect, docs_opinion, labels_opinion

def data_process(x, y, vocab_path, label_list, max_seq_len):
    with open(vocab_path, 'r', encoding = 'utf-8') as f:
        vocabs = f.readlines()
    vocabs = [vocab.strip() for vocab in vocabs]

    vocabs_dict = {}
    for idx, vocab in enumerate(vocabs):
        vocabs_dict[vocab] = idx

    label_dict = {}
    for idx, label in enumerate(label_list):
        label_dict[label] = idx + 1

    docs = []
    labels = []
    for idx, doc in enumerate(x):
        doc_id = []
        for d in doc:
            if vocabs_dict.get(d.lower()) == None:
                if vocabs_dict.get(d) == None:
                    doc_id.append(vocabs_dict['<unk>'])
                else:
                    doc_id.append(vocabs_dict[d])
            else:
                doc_id.append(vocabs_dict[d.lower()])

        label_id = []
        for l in y[idx]:
            label_id.append(label_dict[l])

        if len(doc_id) > max_seq_len:
            print('over max_seq_len')
            doc_id = doc_id[:max_seq_len]
            label_id = label_id[:max_seq_len]
        else:
            for i in range(max_seq_len - len(doc_id)):
                doc_id.append(vocabs_dict['<pad>'])
                label_id.append(0)
        docs.append(doc_id)
        labels.append(label_id)
    return docs, labels

def get_batches(batch_size, x, y, is_random=True):
    if is_random:
        x = np.array(x)
        y = np.array(y)
        shuffle_idx = np.random.permutation(np.arange(len(x)))
        x = x[shuffle_idx]
        y = y[shuffle_idx]

    batches_x = []
    batches_y = []
    for idx in range(int(len(x) / batch_size)):
        batch_x = x[idx * batch_size : (idx + 1) * batch_size]
        batch_y = y[idx * batch_size : (idx + 1) * batch_size]
        batches_x.append(batch_x)
        batches_y.append(batch_y)

    if len(x) % batch_size != 0:
        batches_x.append(x[-(len(x) % batch_size) : ])
        batches_y.append(y[-(len(y) % batch_size) : ])
    
    return batches_x, batches_y

def get_predict_batches(batch_size, x):
    batches_x = []
    for idx in range(int(len(x) / batch_size)):
        batch_x = x[idx * batch_size : (idx + 1) * batch_size]
        batches_x.append(batch_x)
    if len(x) % batch_size != 0:
        batches_x.append(x[-(len(x) % batch_size) : ])
    
    return batches_x