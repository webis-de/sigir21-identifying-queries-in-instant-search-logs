from .temporal import *
from .simple import *
from .similarity import *

feature_template = [
    (calc_time_gap, {}),
    (cosine_similarity_coefficient, {"n" : 3,}),
    (jaccard_similarity_coefficient_ngram, {"n":3}),
    (jaccard_similarity_coefficient_words, {}),
    (modified_jaccard_similarity_coefficient_ngram_short, {"n": 3}),
    (modified_jaccard_similarity_coefficient_ngram_long, {"n": 3}),
    (jaro_winkler, {}),
    (is_prefix, {}),
    (is_prefix_mirrored, {}),
    (is_suffix, {}),
    (is_suffix_mirrored, {}),
    (contains, {}),
    (is_contained, {}),
    (overlap_terms, {}),
    (overlap_characters, {}),
    (length_difference, {}),
    (length_ratio, {}),
    (first_example_query, {}),
    (second_example_query, {}),
    (keeps_operator, {}),
    (first_previous_occurred, {}),
    (second_previous_occured, {}),
]

template_time = [ (calc_time_gap, {}),]

template_similarity_cosine = [(cosine_similarity_coefficient, {"n" : 3, })]
template_similarity_cosine_time = [(cosine_similarity_coefficient, {"n" : 3, }),
                                   (calc_time_gap, {})]

template_similarity_jaccard = [(jaccard_similarity_coefficient_ngram, {"n" : 3,})]
template_similarity_jaccard_time = [(jaccard_similarity_coefficient_ngram, {"n" : 3,}),
                                    (calc_time_gap, {})]

template_length_difference_time = [(length_difference, {}),
                                    (calc_time_gap, {}),]

template_levenshtein = [(normalized_levenshtein_distance, {})]
template_levenshtein_time = [(normalized_levenshtein_distance, {}),
                             (calc_time_gap, {})]