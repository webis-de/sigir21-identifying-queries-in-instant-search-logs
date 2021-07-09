import functools
from timeit import timeit

import numpy as np


def execute_rules(pair, rule_set):
    """execute the rules out of the rule set for the given pair until their is a confident decision"""
    for rule in rule_set:
        result = rule[0](pair, **rule[1])
        if result != -1:
            return result
    return -1


def execute_rule_pipeline(pairs, rule_set):
    """run the rules out of the rule set for all pairs"""
    predicted_labels = np.zeros((len(pairs),))
    true_labels = np.zeros((len(pairs),))
    for i in range(len(pairs)):
        true_labels[i] = pairs[i][0].boundary
        predicted_labels[i] = execute_rules(pairs[i], rule_set)
    return predicted_labels, true_labels
