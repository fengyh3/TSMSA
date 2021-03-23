
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

def evaluate(predicts, labels, label_types):
    aspect_golden_counter = 0
    opinion_golden_counter = 0
    aspect_predict_counter = 0
    opinion_predict_counter = 0
    aspect_correct_counter = 0
    opinion_correct_counter = 0
    for idx, label in enumerate(labels):
        for l in label:
            if l == 'B':
                if label_types[idx] == 0:
                    aspect_golden_counter += 1
                else:
                    opinion_golden_counter += 1

    for idx, predict in enumerate(predicts):
        for pred in predict:
            if pred == 'B':
                if label_types[idx] == 0:
                    aspect_predict_counter += 1
                else:
                    opinion_predict_counter += 1

    labels_pairs = find_boundry(labels)
    predicts_pairs = find_boundry(predicts)

    for idx in range(len(labels_pairs)):
        l_pair = labels_pairs[idx]
        p_pair = predicts_pairs[idx]
        for pair in p_pair:
            if pair in l_pair:
                if label_types[idx] == 0:
                    aspect_correct_counter += 1
                else:
                    opinion_correct_counter += 1

    
    if aspect_predict_counter == 0:
        aspect_predict_counter = 1
    if opinion_predict_counter == 0:
        opinion_predict_counter = 1
    aspect_precision = aspect_correct_counter / aspect_predict_counter
    aspect_recall = aspect_correct_counter / aspect_golden_counter
    aspect_f1 = (2 * aspect_precision * aspect_recall) / (aspect_precision + aspect_recall + 1e-10)
    print('aspect token-level: \nprecision: %s, recall: %s, f1: %s' % (aspect_precision, aspect_recall, aspect_f1))
    
    opinion_precision = opinion_correct_counter / opinion_predict_counter
    opinion_recall = opinion_correct_counter / opinion_golden_counter
    opinion_f1 = (2 * opinion_precision * opinion_recall) / (opinion_precision + opinion_recall + 1e-10)
    print('opinion token-level: \nprecision: %s, recall: %s, f1: %s' % (opinion_precision, opinion_recall, opinion_f1))
    return opinion_f1
'''

def find_boundry(labels, label_types):
    pairs = []
    for idx, label in enumerate(labels):
        flag = False
        start = []
        end = []
        pair = []
        if label_types[idx] == 1:
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
            if pair is not None:
                pairs.append(pair)
        else:
            for idx in range(len(label)):
                if label[idx] == 'B-ASP':
                    if flag:
                        end.append(idx)
                    start.append(idx)
                    flag = True
                elif flag and label[idx] != 'I-ASP':
                    end.append(idx)
                    flag = False
            if flag:
                end.append(len(label))
            for s, e in zip(start, end):
                pair.append((s, e))
            if pair is not None:
                pairs.append(pair)
    return pairs

def evaluate(predicts, labels, label_types):
    aspect_golden_counter = 0
    opinion_golden_counter = 0
    aspect_predict_counter = 0
    opinion_predict_counter = 0
    aspect_correct_counter = 0
    opinion_correct_counter = 0
    for idx, label in enumerate(labels):
        for l in label:
            if l == 'B-ASP':
                aspect_golden_counter += 1
                continue
            if l == 'B':
                opinion_golden_counter += 1

    for idx, predict in enumerate(predicts):
        for pred in predict:
            if pred == 'B-ASP':
                aspect_predict_counter += 1
                continue
            if pred == 'B':
                opinion_predict_counter += 1

    labels_pairs = find_boundry(labels, label_types)
    predicts_pairs = find_boundry(predicts, label_types)

    for idx in range(len(labels_pairs)):
        l_pair = labels_pairs[idx]
        p_pair = predicts_pairs[idx]
        for pair in p_pair:
            if pair in l_pair:
                if label_types[idx] == 0:
                    aspect_correct_counter += 1
                else:
                    opinion_correct_counter += 1

    
    aspect_precision = aspect_correct_counter / aspect_predict_counter
    aspect_recall = aspect_correct_counter / aspect_golden_counter
    aspect_f1 = (2 * aspect_precision * aspect_recall) / (aspect_precision + aspect_recall + 1e-10)
    print('aspect token-level: \nprecision: %s, recall: %s, f1: %s' % (aspect_precision, aspect_recall, aspect_f1))
    
    opinion_precision = opinion_correct_counter / opinion_predict_counter
    opinion_recall = opinion_correct_counter / opinion_golden_counter
    opinion_f1 = (2 * opinion_precision * opinion_recall) / (opinion_precision + opinion_recall + 1e-10)
    print('opinion token-level: \nprecision: %s, recall: %s, f1: %s' % (opinion_precision, opinion_recall, opinion_f1))

    return opinion_f1
'''