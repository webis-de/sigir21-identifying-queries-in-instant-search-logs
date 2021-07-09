import numpy as np
from sklearn.metrics import recall_score, confusion_matrix, accuracy_score, precision_score, fbeta_score


def relaxed_precision_score(y_true, y_pred, relax_factor=0.5):
    """a relaxed precision score which weights off-by-one errors with the relax_factor"""
    true_positive = np.zeros_like(y_true)
    MAX = y_true.shape[0]
    MIN = 0
    for i in list(np.nonzero(y_pred)[0]):
        if y_true[i] == 1:
            true_positive[i] = 1
            continue
        if i - 1 >= MIN:
            if y_true[i-1] == 1:
                true_positive[i] += relax_factor * 1
        if i + 1 < MAX:
            if y_true[i+1] == 1:
                true_positive[i] += relax_factor * 1
    return np.sum(true_positive)/np.sum(y_pred)


def tp(y_true, y_pred):
    """returns the amount of true positives"""
    try:
        tn_, fp_, fn_, tp_ = confusion_matrix(y_true, y_pred).ravel()
    except ValueError:
        tp_ = confusion_matrix(y_true, y_pred)[0][0]
        fp_ = fn_ = tn_ = 0
    return tp_


def tn(y_true, y_pred):
    """returns the amount of true negatives"""
    try:
        tn_, fp_, fn_, tp_ = confusion_matrix(y_true, y_pred).ravel()
    except ValueError:
        tp_ = fp_ = fn_ = tn_ = 0
    return tn_


def fp(y_true, y_pred):
    """returns the amount of false positives"""
    try:
        tn_, fp_, fn_, tp_ = confusion_matrix(y_true, y_pred).ravel()
    except ValueError:
        tp_ = fp_ = fn_ = tn_ = 0
    return fp_


def fn(y_true, y_pred):
    """returns the amount of false negatives"""
    try:
        tn_, fp_, fn_, tp_ = confusion_matrix(y_true, y_pred).ravel()
    except ValueError:
        tp_ = fp_ = fn_ = tn_ = 0
    return fn_


def evaluate_model(y_true, y_pred):
    """evaluates the important metrics and print them out"""
    print(f"Results: ACC = {accuracy_score(y_true, y_pred):.2f} "
          f"PRE = {precision_score(y_true, y_pred):.2f} "
          f"REC = {recall_score(y_true, y_pred):.2f} "
          f"F2 = {fbeta_score(y_true, y_pred, beta=2):.2f} "
          f"TP = {tp(y_true, y_pred)} TN = {tn(y_true, y_pred)} "
          f"FP = {fp(y_true, y_pred)} FN = {fn(y_true, y_pred)}")
