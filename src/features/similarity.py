import math
import subprocess

from similarity.levenshtein import Levenshtein
from similarity.longest_common_subsequence import LongestCommonSubsequence
from similarity.jarowinkler import JaroWinkler

from .featureextraction import preprocess_search_string


def cosine_similarity_coefficient(interaction, next_interaction, n):
    """computes standard cosine similarity coefficient for n-grams"""
    string1 = preprocess_search_string(interaction.interaction)
    string2 = preprocess_search_string(next_interaction.interaction)

    while len(string1) < n:
        string1 += " "

    while len(string2) < n:
        string2 += " "

    string1_ngrams = [string1[i:i + n] for i in range(len(string1) - n + 1)]
    string2_ngrams = [string2[i:i + n] for i in range(len(string2) - n + 1)]
    string1_ngrams_count = {}
    string2_ngrams_count = {}
    for ngram in string1_ngrams:
        if ngram in string1_ngrams_count.keys():
            string1_ngrams_count[ngram] += 1
        else:
            string1_ngrams_count[ngram] = 1

    for ngram in string2_ngrams:
        if ngram in string2_ngrams_count.keys():
            string2_ngrams_count[ngram] += 1
        else:
            string2_ngrams_count[ngram] = 1

    if len(string1_ngrams_count.keys()) < len(string2_ngrams_count):
        temp = string2_ngrams_count
        string2_ngrams_count = string1_ngrams_count
        string1_ngrams_count = temp

    scalar_product = 0
    for ngram, count in string1_ngrams_count.items():
        if ngram in string2_ngrams_count.keys():
            scalar_product += 1.0 * count * string2_ngrams_count[ngram]

    string1_norm = 0
    string2_norm = 0

    for ngram, count in string1_ngrams_count.items():
        string1_norm += 1.0 * count ** 2

    for ngram, count in string2_ngrams_count.items():
        string2_norm += 1.0 * count ** 2

    string1_norm = math.sqrt(string1_norm)
    string2_norm = math.sqrt(string2_norm)

    return scalar_product / (string1_norm * string2_norm)


def jaccard_similarity_coefficient_ngram(interaction, next_interaction, n):
    """computes jaccard similarity coefficient for ngrams"""
    string1 = preprocess_search_string(interaction.interaction)
    string2 = preprocess_search_string(next_interaction.interaction)

    string1_ngrams = [string1[i:i + n] for i in range(len(string1) - n + 1)]
    string2_ngrams = [string2[i:i + n] for i in range(len(string2) - n + 1)]

    set_1 = set(string1_ngrams)
    set_2 = set(string2_ngrams)
    if not len(set_1) or not len(set_2):
        return 0

    return len(set_1.intersection(set_2)) / float(len(set_1.union(set_2)))


def modified_jaccard_similarity_coefficient_ngram_short(interaction, next_interaction, n):
    """computes a modified jaccard similiarity coefficient by dividingy only through the smaller set"""
    string1 = preprocess_search_string(interaction.interaction)
    string2 = preprocess_search_string(next_interaction.interaction)

    string1_ngrams = [string1[i:i + n] for i in range(len(string1) - n + 1)]
    string2_ngrams = [string2[i:i + n] for i in range(len(string2) - n + 1)]

    set_1 = set(string1_ngrams)
    set_2 = set(string2_ngrams)
    if not len(set_1) or not len(set_2):
        return 0

    return len(set_1.intersection(set_2)) / float(min(len(set_1), len(set_2)))


def modified_jaccard_similarity_coefficient_ngram_long(interaction, next_interaction, n):
    """computes a modified jaccard similiarity coefficient by dividingy only through the larger set"""
    string1 = preprocess_search_string(interaction.interaction)
    string2 = preprocess_search_string(next_interaction.interaction)

    string1_ngrams = [string1[i:i + n] for i in range(len(string1) - n + 1)]
    string2_ngrams = [string2[i:i + n] for i in range(len(string2) - n + 1)]

    set_1 = set(string1_ngrams)
    set_2 = set(string2_ngrams)
    if not len(set_1) or not len(set_2):
        return 0

    return len(set_1.intersection(set_2)) / float(max(len(set_1), len(set_2)))


def jaccard_similarity_coefficient_words(interaction, next_interaction):
    """computes jaccard similarity coefficient for words"""
    string1 = preprocess_search_string(interaction.interaction)
    string2 = preprocess_search_string(next_interaction.interaction)

    string1_words = set(string1.split())
    string2_words = set(string2.split())

    if not len(string1_words) or not len(string2_words):
        return 0

    return len(string1_words.intersection(string2_words)) / float(len(string1_words.union(string2_words)))


def rouge_lcs(interaction, next_interaction, beta=1):
    """computes rouge lcs coefficient"""
    lcs = LongestCommonSubsequence().length(preprocess_search_string(interaction.interaction), preprocess_search_string(next_interaction.interaction))
    if lcs == 0:
        return 0

    rec = lcs/float(len(preprocess_search_string(interaction.interaction)))
    pre = lcs/float(len(preprocess_search_string(next_interaction.interaction)))

    return ((1 + beta * beta) * rec * pre)/float((rec + beta * beta * pre))


def rouge_lcs_rev(interaction, next_interaction, beta=1):
    """computes the rouge lcs coefficient but interaction and next_interaction reveresed"""
    return rouge_lcs(next_interaction, interaction, beta)


def jaro_winkler(interaction, next_interaction):
    """computes the jaro winkler coefficient"""
    return JaroWinkler().similarity(preprocess_search_string(interaction.interaction), preprocess_search_string(next_interaction.interaction))


def levenshtein_distance(interaction, next_interaction):
    """computes levenshtein edit distance"""
    levenshtein = Levenshtein()
    return levenshtein.distance(preprocess_search_string(interaction.interaction), preprocess_search_string(next_interaction.interaction))


def normalized_levenshtein_distance(interaction, next_interaction):
    """computes normalized levensthein edit distance"""
    return levenshtein_distance(interaction, next_interaction) / max(1, len(preprocess_search_string(interaction.interaction)), len(preprocess_search_string(next_interaction.interaction)))


def esa(interaction, next_interaction):
    """computes the esa semantic similarity with an external java program"""
    esa_string = subprocess.run(['java', '-jar', '/mnt/data/netspeakdata/esa/esa_step.jar', interaction.interaction, next_interaction.interaction],
                                capture_output=True).stdout
    esa_score = 0
    if esa_string != b'':
        esa_score = float(esa_string)
    return esa_score