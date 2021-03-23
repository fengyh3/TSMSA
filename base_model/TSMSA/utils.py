import numpy as np
import matplotlib.pyplot as plt
import tokenization

def build_vocab(train_doc, test_doc, vocab_path, lower = True):
    vocab = []
    for doc in train_doc:
        for word in doc:
            word = word.lower()
            if word not in vocab:
                vocab.append(word)
    for doc in test_doc:
        for word in doc:
            word = word.lower()
            if word not in vocab:
                vocab.append(word)

    if '‖' not in vocab:
        vocab.append('‖')
    vocab.insert(0, '<pad>')
    vocab.insert(0, '<unk>')
    with open(vocab_path, 'w', encoding = 'utf-8') as f:
        for v in vocab:
            f.write(v + '\n')
    return vocab

def read_data(path):
    with open(path, 'r', encoding = 'utf-8') as f:
        data = f.readlines()
    data = [d.strip() for d in data]

    docs = []
    labels = []
    doc = []
    label = []
    for line in data:
        if line == '1':
            continue
        if line == '':
            docs.append(doc)
            labels.append(label)
            doc = []
            label = []
            continue
        doc.append(line.split(' ')[0])
        label.append(line.split(' ')[1])

    return docs, labels

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

def data_process_for_BERT(x, y, vocab_path, label_list, max_seq_length, lower=True):
    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_path, do_lower_case=lower)
    label_map = {}
    for idx, label in enumerate(label_list):
        label_map[label] = idx + 1
    input_ids = []
    label_ids = []
    for idx, doc in enumerate(x):
        #print(i)
        tokens = []
        labels = []
        input_id = []
        label_id = []
        for idx1, word in enumerate(doc):
            token = tokenizer.tokenize(word)
            label_1 = y[idx][idx1]
            tokens.extend(token)

            for m in range(len(token)):
                if m == 0:
                    labels.append(label_1)
                else:
                    labels.append("X")
        
        if len(tokens) > max_seq_length:
            tokens = tokens[0:max_seq_length]
            labels = labels[0:max_seq_length]

        ntokens = []
        
        for i, token in enumerate(tokens):
            ntokens.append(token)
            label_id.append(label_map[labels[i]])

        input_id = tokenizer.convert_tokens_to_ids(ntokens)

        while len(input_id) < max_seq_length:
            input_id.append(0)
            label_id.append(0)

        input_ids.append(input_id)
        label_ids.append(label_id)

    return input_ids, label_ids

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

def plot(x, Y, file_name):
    plt.rcParams['font.sans-serif'] = ['Arial']  # 如果要显示中文字体,则在此处设为：SimHei
    plt.rcParams['axes.unicode_minus'] = False  # 显示负号

    #x = np.array([1, 2, 3, 4, 5, 6])

    VGG_supervised = np.array([2.9749694, 3.9357018, 4.7440844, 6.482254, 8.720203, 13.687582])
    VGG_unsupervised = np.array([2.1044724, 2.9757383, 3.7754183, 5.686206, 8.367847, 14.144531])
    ourNetwork = np.array([2.0205495, 2.6509762, 3.1876223, 4.380781, 6.004548, 9.9298])

    # label在图示(legend)中显示。若为数学公式,则最好在字符串前后添加"$"符号
    # color：b:blue、g:green、r:red、c:cyan、m:magenta、y:yellow、k:black、w:white、、、
    # 线型：-  --   -.  :    ,
    # marker：.  ,   o   v    <    *    +    1
    plt.figure(figsize=(10, 5))
    plt.grid(linestyle="--")  # 设置背景网格线为虚线
    ax = plt.gca()
    ax.spines['top'].set_visible(False)  # 去掉上边框
    ax.spines['right'].set_visible(False)  # 去掉右边框


    plt.plot(x, VGG_supervised, marker='o', color="blue", label="VGG-style Supervised Network", linewidth=1.5)
    plt.plot(x, VGG_unsupervised, marker='o', color="green", label="VGG-style Unsupervised Network", linewidth=1.5)
    plt.plot(x, ourNetwork, marker='o', color="red", label="ShuffleNet-style Network", linewidth=1.5)

    group_labels = ['Top 0-5%', 'Top 5-10%', 'Top 10-20%', 'Top 20-50%', 'Top 50-70%', ' Top 70-100%']  # x轴刻度的标识
    plt.xticks(x, group_labels, fontsize=12, fontweight='bold')  # 默认字体大小为10
    plt.yticks(fontsize=12, fontweight='bold')
    # plt.title("example", fontsize=12, fontweight='bold')  # 默认字体大小为12
    plt.xlabel("Performance Percentile", fontsize=13, fontweight='bold')
    plt.ylabel("4pt-Homography RMSE", fontsize=13, fontweight='bold')
    plt.xlim(0.9, 6.1)  # 设置x轴的范围
    plt.ylim(1.5, 16)

    # plt.legend()          #显示各曲线的图例
    plt.legend(loc=0, numpoints=1)
    leg = plt.gca().get_legend()
    ltext = leg.get_texts()
    plt.setp(ltext, fontsize=12, fontweight='bold')  # 设置图例字体的大小和粗细

    plt.savefig('./' + file_name + '.eps', format='eps')  # 建议保存为svg格式,再用inkscape转为矢量图emf后插入word中
    plt.show()