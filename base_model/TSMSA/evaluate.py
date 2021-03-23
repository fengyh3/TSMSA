def find_boundry(labels):
    pairs = []
    for label in labels:
        flag = False
        start = []
        end = []
        pair = []
        for idx in range(len(label)):
            if label[idx] == 'B':
                if flag:
                    end.append(idx)
                start.append(idx)
                flag = True
            elif flag and label[idx] != 'I':
                end.append(idx)
                flag = False
        if flag:
            end.append(len(label))
        for s, e in zip(start, end):
            pair.append((s, e))
        pairs.append(pair)
    return pairs

def evaluate(predicts, labels):
    opinion_golden_counter = 0
    opinion_predict_counter = 0
    opinion_correct_counter = 0
    for idx, label in enumerate(labels):
        for l in label:
            if l == 'B':
                opinion_golden_counter += 1

    for idx, predict in enumerate(predicts):
        for pred in predict:
            if pred == 'B':
                opinion_predict_counter += 1

    labels_pairs = find_boundry(labels)
    predicts_pairs = find_boundry(predicts)

    for idx in range(len(labels_pairs)):
        l_pair = labels_pairs[idx]
        p_pair = predicts_pairs[idx]
        for pair in p_pair:
            if pair in l_pair:
                opinion_correct_counter += 1

    opinion_precision = opinion_correct_counter / (opinion_predict_counter + 1)
    opinion_recall = opinion_correct_counter / (opinion_golden_counter + 1)
    opinion_f1 = (2 * opinion_precision * opinion_recall) / (opinion_precision + opinion_recall + 1e-10)
    print('opinion token-level: \nprecision: %s, recall: %s, f1: %s' % (opinion_precision, opinion_recall, opinion_f1))
    return opinion_f1