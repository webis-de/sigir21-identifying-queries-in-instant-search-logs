from similarity.longest_common_subsequence import LongestCommonSubsequence

from . import preprocess_search_string


def starts_with_same_character(interaction, next_interaction):
    """returns true if both interactions start_time with same character"""
    try:
        return preprocess_search_string(interaction.interaction[0]) == preprocess_search_string(next_interaction.interaction[0])
    except IndexError:
        return False


def contains(interaction, next_interaction):
    """returns true if the first interaction contains the second"""
    return is_contained(next_interaction, interaction)


def is_prefix(interaction, next_interaction):
    """returns true if the first interaction is prefix of the second"""
    return preprocess_search_string(next_interaction.interaction).startswith(preprocess_search_string(interaction.interaction))


def is_prefix_mirrored(interaction, next_interaction):
    """returns true if the seconds interaction is the prefix of the first"""
    return is_prefix(next_interaction, interaction)


def is_contained(interaction, next_interaction):
    """returns true if the first interaction is contained in the second"""
    return preprocess_search_string(interaction.interaction) in preprocess_search_string(next_interaction.interaction)


def is_suffix(interaction, next_interaction):
    """returns true if the first interaction is the suffix of the second"""
    return preprocess_search_string(next_interaction.interaction).endswith(preprocess_search_string(interaction.interaction))


def is_suffix_mirrored(interaction, next_interaction):
    """returns true if the second interaction is the suffix of the first"""
    return is_suffix(next_interaction, interaction)


def overlap_terms(interaction, next_interaction):
    """returns the number of terms contained in both interactions"""
    return len(set(preprocess_search_string(interaction.interaction).split()).intersection(set(preprocess_search_string(next_interaction.interaction).split())))


def overlap_characters(interaction, next_interaction):
    """returns the number of characters contained in both interactions"""
    return len(set(preprocess_search_string(interaction.interaction)).intersection(set(preprocess_search_string(next_interaction.interaction))))


def length_difference(interaction, next_interaction):
    """return the length difference between the both interactions"""
    return len(list(preprocess_search_string(interaction.interaction))) - len(list(preprocess_search_string(next_interaction.interaction)))


def length_ratio(interaction, next_interaction):
    """returns the length ratio between both interactions"""
    if len(preprocess_search_string(next_interaction.interaction)) == 0:
        return 0
    return len(preprocess_search_string(interaction.interaction)) / float(len(preprocess_search_string(next_interaction.interaction)))


def length_difference_lcs(interaction, next_interaction):
    lcs = LongestCommonSubsequence()
    return lcs.distance(preprocess_search_string(interaction.interaction), preprocess_search_string(next_interaction.interaction))


def first_example_query(interaction, next_interaction):
    """returns true if the first interaction is one of the example queries"""
    example_queries = ["how to ? this", "see ... works", "it's [ great well ]", "and knows #much", "{ more show me }", "m...d ? g?p", "waiting ? response", "waiting + response", "waiting ? ? response", "waiting * response", "the same [ like as ]", "{ only for members }", "waiting * #response", "waiting ? ? response | waiting ? response"]
    return preprocess_search_string(interaction.interaction) in example_queries


def second_example_query(interaction, next_interaction):
    """returns true if the second interaction is one of the example queries"""
    return first_example_query(next_interaction, interaction)


def keeps_operator(interaction, next_interaction):
    """returns true if the custom netspeak operator is present in both interactions"""
    unary_operators = ["...", "?", "#"]
    binary_operators = [["{", "}"], ["[", "]"]]
    string1 = preprocess_search_string(interaction.interaction)
    string2 = preprocess_search_string(next_interaction.interaction)
    for operator in unary_operators:
        if operator in string1 and operator in string2:
            return True
    for operator in binary_operators:
        if operator[0] in string1 and operator[1] in string1 and operator[0] in string2 and operator[1] in string2:
            return True
    return False


def count_previous_occurrence(data):
    """counts how many times a interaction occurred with the same content before (must called as preprocessing step)"""
    for user in data.keys():
        user_entries = {}
        for entry in data[user]:
            if preprocess_search_string(entry.interaction) in user_entries:
                entry.occurred_before = user_entries[preprocess_search_string(entry.interaction)]
                user_entries[preprocess_search_string(entry.interaction)] += 1
            else:
                user_entries[preprocess_search_string(entry.interaction)] = 1


def first_previous_occurred(interaction, next_interaction):
    """returns how often the first interaction occurred before"""
    return interaction.occurred_before


def second_previous_occured(interaction, next_interaction):
    """returns how often the second interaction occurred before"""
    return next_interaction.occurred_before


def results_overlap(interaction, next_interaction):
    """if available returns true if the netspeak results overlap"""
    if interaction.netspeak_results is None or next_interaction.netspeak_results is None:
        return False
    return not set(interaction.netspeak_results).isdisjoint(set(next_interaction.netspeak_results))