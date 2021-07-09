from .parameters import *
from features.temporal import calc_time_gap
from features.simple import contains, is_contained, is_prefix, results_overlap
from features import normalized_levenshtein_distance
from features.similarity import cosine_similarity_coefficient, jaccard_similarity_coefficient_ngram, esa


def time_step(pair, upper_time_threshold=UPPER_TIME_THRESHOLD):
    """split pairs if the time gap between exceeds the threshold"""
    time_gap = calc_time_gap(pair[0], pair[1])
    pair[0].time_difference_next_entry = time_gap
    if time_gap > upper_time_threshold:
        return 1
    else:
        return -1


def containment_step(pair):
    """merges if one entry is contained in the other"""
    if is_contained(pair[0], pair[1]) or contains(pair[0], pair[1]):
        return 0
    else:
        return -1


def containment_step_with_time(pair, lower_time_containment_threshold=LOWER_TIME_CONTAINMENT_THRESHOLD):
    """merges if one entry is contained in the other and the time differences lies below the threshold"""
    if is_contained(pair[0], pair[1]) and pair[0].time_difference_next_entry < lower_time_containment_threshold:
        return 0
    else:
        return -1


def prefix_step(pair, **kwargs):
    """merges if the first pair entry is prefix of the second pair entry"""
    if is_prefix(pair[0], pair[1]):
        return 0
    else:
        return -1


def prefix_step_with_time(pair, lower_time_pre_suf_threshold=LOWER_TIME_PRE_SUF_THRESHOLD):
    """merges if the first pair entry is prefix of the second pair entry and the time differences lies below the threshold"""
    if (is_prefix(pair[0], pair[1])) and pair[0].time_difference_next_entry < lower_time_pre_suf_threshold:
        return 0
    else:
        return -1


def ngram_similarity_step(pair, n=3, similarity_threshold=None, similarity_time_threshold=None):
    """merges if the cosine similarity between both entries lies above the threshold and the time gap below the threshold"""
    if similarity_threshold is None:
        similarity_threshold = SIMILARITY_THRESHOLD[n]
    if similarity_time_threshold is None:
        similarity_time_threshold = SIMILARITY_TIME_THRESHOLD[n]
    if len(pair[0].interaction) < n or len(pair[1].interaction) < n:
        return -1
    if cosine_similarity_coefficient(pair[0], pair[1], n) >= similarity_threshold \
            and calc_time_gap(pair[0], pair[1]) < similarity_time_threshold:
        #print(f"Sim: {calc_ngram_similarity(pair[0], pair[1], n)}: {pair}")
        return 0
    else:
        return -1


def jaccard_similarity_step(pair, n=3, jaccard_threshold=None, jaccard_time_threshold=None):
    """merges if the jaccard similarity between both entries lies above the threshold and the time gap below the threshold"""
    if jaccard_threshold is None:
        jaccard_threshold = JACCARD_THRESHOLD[n]
    if jaccard_time_threshold is None:
        jaccard_time_threshold = JACCARD_TIME_THRESHOLD[n]
    if len(pair[0].interaction) < n or len(pair[1].interaction) < n:
        return -1
    if jaccard_similarity_coefficient_ngram(pair[0], pair[1], n) >= jaccard_threshold \
            and pair[0].time_difference_next_entry < jaccard_time_threshold:
        return 0
    else:
        return -1


def jaccard_dissimilarity_step(pair, n=3, jaccard_threshold=None, jaccard_time_threshold=None):
    """splits if the jaccard similarity between both entries lies below the threshold and the time gap exceeds the threshold"""
    if jaccard_threshold is None:
        jaccard_threshold = JACCARD_THRESHOLD_BORDER[n]
    if jaccard_time_threshold is None:
        jaccard_time_threshold = JACCARD_TIME_THRESHOLD_BORDER[n]
    if len(pair[0].interaction) <= n or len(pair[1].interaction) <= n:
        return -1
    if jaccard_similarity_coefficient_ngram(pair[0], pair[1], n) <= jaccard_threshold \
            and pair[0].time_difference_next_entry >= jaccard_time_threshold:
        return 1
    else:
        return -1


def levenshtein_distance_step(pair, levenshtein_threshold=LEVENSHTEIN_THRESHOLD, levenshtein_time_threshold=LEVENSHTEIN_TIME_THRESHOLD):
    """merge if the normalized edit distance lies below the threshold and the time gap between both is small enough"""
    if normalized_levenshtein_distance(pair[0], pair[1]) <= levenshtein_threshold\
            and calc_time_gap(pair[0], pair[1]) < levenshtein_time_threshold:
        return 0
    else:
        return -1


def edit_decision(pair, edit_decision_threshold=EDIT_DECISION_THRESHOLD):
    """merge if the edit distance between both entries is below of the threshold; split otherwise"""
    if normalized_levenshtein_distance(pair[0], pair[1]) <= edit_decision_threshold:
        return 0
    else:
        return 1


def geometric_method(pair, n, f_time_threshold=0.6, time_max=5400., f_lex_threshold=0.15):
    """merge and split as described by Hagen et al. (2013)"""
    if len(pair[0].interaction) <= n or len(pair[1].interaction) <= n:
        return -1
    f_lex = cosine_similarity_coefficient(pair[0], pair[1], n)
    if f_lex > f_lex_threshold:
        return 0
    f_time = 1 - (pair[0].time_difference_next_entry/time_max)
    if f_time < f_time_threshold:
        return 1
    return -1


def esa_step(pair, f_esa_threshold=0.28):
    """merge if the esa score exceeds the threshold"""
    esa_score = esa(pair[0], pair[1])
    if esa_score >= f_esa_threshold:
        return 0
    return -1


def results_overlap_step(pair):
    """merge if results overlap"""
    if results_overlap(pair[0], pair[1]):
        return 0
    else:
        return -1


def split_remainder(pair):
    """always split"""
    return 1


def merge_remainder(pair):
    """always merge"""
    return 0