import datetime
import math
import re

from nltk.stem import WordNetLemmatizer
from sklearn.metrics import f1_score, precision_score, recall_score

from .bleu import compute_bleu
from .rouge import rouge

###############################################################################
# Evaluation, data-postprocessing functions
###############################################################################


def rouge_score(references, generated):
    """both are a list of strings"""
    score = rouge(generated, references)
    rouge_s = {k: (v * 100) for (k, v) in score.items()}
    """
    "rouge_1/f_score": rouge_1_f,
    "rouge_1/r_score": rouge_1_r,
    "rouge_1/p_score": rouge_1_p,
    "rouge_2/f_score": rouge_2_f,
    "rouge_2/r_score": rouge_2_r,
    "rouge_2/p_score": rouge_2_p,
    "rouge_l/f_score": rouge_l_f,
    "rouge_l/r_score": rouge_l_r,
    "rouge_l/p_score": rouge_l_p,
    """
    return rouge_s


def bleu_score(references, generated, n_gram=4, smooth=False):
    """a list of lists of tokens"""
    formatted_ref = [[ref] for ref in references]
    bleu_s, _, _, _, _, _ = compute_bleu(formatted_ref, generated, n_gram, smooth)
    return bleu_s * 100


def two_seq_same(sa, sb):
    if len(sa) != len(sb):
        return False
    for wa, wb in zip(sa, sb):
        if wa != wb:
            return False
    return True


def unique_sentence_percent(sequence_batch):
    unique_seq = []
    for seq in sequence_batch:
        count = 0
        for uni_seq in unique_seq:
            if two_seq_same(seq, uni_seq):
                count += 1
                break
        if count == 0:
            unique_seq.append(seq)

    return len(unique_seq) / len(sequence_batch), len(unique_seq)


def getstem(word, lemmatizer=None):
    return lemmatizer.lemmatize(word)

def feature_detect(seq_batch, feature_set):
    lemmatizer = WordNetLemmatizer()
    feature_batch = []
    for ids in seq_batch:
        feature_list = []
        for i in ids:
            i = getstem(i, lemmatizer)
            if i in feature_set:
                feature_list.append(i)

        feature_batch.append(set(feature_list))

    return feature_batch


def feature_matching_ratio(feature_batch, test_feature):
    count = 0
    for fea_set, fea in zip(feature_batch, test_feature):
        if isinstance(fea, list):
            for f in fea:  # a list of features
                if f in fea_set:
                    count += 1
        else:  # single feature
            if fea in fea_set:
                count += 1
    return count / len(feature_batch)


def feature_coverage_ratio(feature_batch, feature_set):
    features = set()
    for fb in feature_batch:
        features = features | fb

    return len(features) / len(feature_set)


def feature_diversity(feature_batch):
    list_len = len(feature_batch)

    total_count = 0
    for i, x in enumerate(feature_batch):
        for j in range(i + 1, list_len):
            y = feature_batch[j]
            total_count += len(x & y)

    denominator = list_len * (list_len - 1) / 2
    return total_count / denominator


def mean_absolute_error(predicted, max_r, min_r, mae=True):
    total = 0
    for r, p in predicted:
        p = max(p, max_r)
        p = min(p, min_r)
        sub = p - r
        if mae:
            total += abs(sub)
        else:
            total += sub**2

    return total / len(predicted)


def root_mean_square_error(predicted, max_r, min_r):
    mse = mean_absolute_error(predicted, max_r, min_r, False)
    return math.sqrt(mse)


def class_metrics(predicted,
                  class_num:int=3, average:str='macro'):

    # print("predicted", predicted)

    if class_num == 3:
        interval_list = [-1, -0.2, 0, 0.2, 1]
    else: # class_num == 2
        interval_list = [-1, 0, 1]

    def classify(x):
        for idx, interval in enumerate(interval_list):
            if x <= interval:
                return idx
        return len(interval_list)-1

    r_labels, p_labels = [], []
    for r, p in predicted:
        r_labels.append(classify(r))
        p_labels.append(classify(p))
    f1 = f1_score(r_labels, p_labels, average=average)
    precision = precision_score(r_labels, p_labels, average=average)
    recall = recall_score(r_labels, p_labels, average=average)
    acc = sum([1 for r, p in zip(r_labels, p_labels) if r == p]) / len(r_labels)
    return f1, precision, recall, acc



def now_time():
    return "[" + datetime.datetime.now().strftime("%Y-%m-%d %H:%M") + "]: "


def postprocessing(string):
    """
    adopted from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub("'s", " 's", string)
    string = re.sub("'m", " 'm", string)
    string = re.sub("'ve", " 've", string)
    string = re.sub("n't", " n't", string)
    string = re.sub("'re", " 're", string)
    string = re.sub("'d", " 'd", string)
    string = re.sub("'ll", " 'll", string)
    string = re.sub("\(", " ( ", string)
    string = re.sub("\)", " ) ", string)
    string = re.sub(",+", " , ", string)
    string = re.sub(":+", " , ", string)
    string = re.sub(";+", " . ", string)
    string = re.sub("\.+", " . ", string)
    string = re.sub("!+", " ! ", string)
    string = re.sub("\?+", " ? ", string)
    string = re.sub(" +", " ", string).strip()
    return string


def ids2tokens(ids, tokenizer, eos):
    text = tokenizer.decode(ids)
    text = postprocessing(text)  # process punctuations: "good!" -> "good !"
    tokens = []
    for token in text.split():
        if token == eos:
            break
        tokens.append(token)
    return tokens
