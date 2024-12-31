from sklearn.metrics import confusion_matrix
import numpy as np
from Common.Struct import Struct
import math
from Common.bcolors import bcolors
from Common.Result import Result
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from imblearn.metrics import geometric_mean_score


def report_bin_classification(y_true, y_pred, probs=None, percent=True, inc_labels=True):
    res = Result()
    res.conf_matrix = confusion_matrix(y_true, y_pred)
    TN, FP, FN, TP = res.conf_matrix.ravel()
    coeff = 100 if percent else 1
    try:
        res.acc = coeff * (TP + TN) / float(TP + TN + FP + FN)
        eps = 1e-10 if (TP + FN) == 0 else 0.
        res.sensitivity = coeff * (TP / float(TP + FN + eps))
        eps = 1e-10 if (TN + FP) == 0 else 0.
        res.specificity = coeff * (TN / float(TN + FP + eps))
        eps = 1e-10 if (TP + FP) == 0 else 0.
        res.precision = coeff * (TP / float(TP + FP + eps))
        eps = 1e-10 if (res.precision + res.sensitivity) == 0 else 0.
        res.f_score = 2 * ((res.precision * res.sensitivity) / (res.precision + res.sensitivity + eps))
    except:
        print('Exception occured in computing classification metrics!!!')

    # G - mean is the squared root of the product of the sensitivity and specificity.
    res.g_mean = np.sqrt(res.sensitivity * res.specificity)
    if probs is not None:
        if len(probs.shape) > 1:
            probs = probs[:, 1]
            res.roc_auc = roc_auc_score(y_true, probs) * 100

    if inc_labels:
        res.y_true, res.y_pred = y_true, y_pred

    return res

def report_multi_classification(y_true, y_pred, probs=None, percent=True, inc_labels=True):
    result = classification_report(y_true, y_pred, zero_division=0, output_dict=True)
    if percent:
        import numbers
        for k in result.keys():
            if isinstance(result[k], numbers.Number):
                result[k] *= 100
                continue
            for m in result[k].keys():
                if m.lower() == 'support':
                    continue
                result[k][m] *= 100

    res = Result()
    res.report = classification_report(y_true, y_pred, zero_division=0, digits=4)
    res.dic = result
    res.acc = result['accuracy']
    res.f_score = result['weighted avg']['f1-score']
    res.precision = result['weighted avg']['precision']
    res.recall = result['weighted avg']['recall']
    res.sensitivity = res.recall
    res.g_mean = geometric_mean_score(y_true, y_pred) * 100
    res.balanced_acc = result['macro avg']['recall']
    res.balanced_precision = result['macro avg']['precision']
    res.balanced_f_score = result['macro avg']['f1-score']
    res.multi_class = True
    res.conf_matrix = confusion_matrix(y_true, y_pred)

    if inc_labels:
        res.y_pred, res.probs, res.y_true = y_pred, probs, y_true
    return res


def print_res(res, color=None, conf_matrix=False):
    if color is None:
      color = bcolors.FAIL

    if hasattr(res, 'report'):  # multiclass
      print(color + res.report + bcolors.ENDC)
      s = ''
      if hasattr(res, 'epoch'):
        s = f'epoch:{res.epoch}, ' + s

      if hasattr(res, 'loss'):
        s += f' loss= {res.loss:.4f}, '

      if hasattr(res, 'g_mean'):
        s += f' g_mean= {res.g_mean:.2f}, '

      print(color + s + bcolors.ENDC)
      return

    s = """accuracy: {:.2f}, f_score: {:.2f}, sensitivity:{:.2f}, precision:{:.2f},
    specificity:{:.2f}, g_mean:{:.2f}"""

    if hasattr(res, 'epoch'):
      s = f'epoch:{res.epoch}, ' + s

    if hasattr(res, 'loss'):
      s = f'loss= {res.loss:.4f}, ' + s

    if hasattr(res, 'roc_auc'):
      s += f', roc_auc= {res.roc_auc:.2f}, '

    print(color + s.format(res.acc, res.f_score, res.sensitivity,
                           res.precision, res.specificity, res.g_mean) + bcolors.ENDC)

    if conf_matrix:
      print('confusion matrix =')
      print(res.conf_matrix)


def main():

    pass


if __name__ == '__main__':
    main()
    print('Congratulations to you!')