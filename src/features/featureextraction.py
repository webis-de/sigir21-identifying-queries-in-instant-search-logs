import numpy as np

from data import make_pairs, labels_for_pairs, generate_data_matrix


def extract_features_compared_to(feature, entries, index, offset, **kwars):
    """Extract feature for log entry at index compared to log entry at index + offset"""
    if index + offset < 0 or index < 0:
        return 0
    elif index + offset >= len(entries) or index >= len(entries):
        return 0
    else:
        return feature(entries[index], entries[index + offset], **kwars)


def build_feature_set(template_set, look_back_count = 3):
    """ Builds a feature set out of a template list.

    :param template_set: list of tuples of the format (function, additional arguments dictionary)
    :param look_back_count: to which entry should we look back
    :return: list of tuples of the format (function, extraction offset, additional arguments dictionary)
    """
    result_set = []
    for i in range(look_back_count):
        for template in template_set:
            result_set.append((template[0], -(i + 1), template[1]))
    return result_set


def extract_features(entry_list, features, entry_index=None):
    """Extract features specified in features for each entry in the entry list if entry_index is None or for the specified entry only"""
    if entry_index is None:
        temp = entry_list
        for i in range(len(entry_list)):
            temp[i].features = extract_features(entry_list, features, i)
        return temp
    else:
        if entry_index >= len(entry_list):
            return 0
        feature_vector = np.zeros(len(features))
        index = 0
        for feature, offset, args in features:
            if offset != 0:
                feature_vector[index] = extract_features_compared_to(feature, entry_list, entry_index, offset, **args)
            else:
                feature_vector[index] = feature(entry_list[entry_index])
            index += 1
        else:
            return feature_vector


def extract_features_dynamic(entry_list, features_template, constrained=None):
    """Extract the features for each entry in the feature list but with dynamic look back to the last split"""
    temp = entry_list
    look_back_count = 1
    temp[0].features = extract_features(entry_list, build_feature_set(features_template, look_back_count), 1)

    next_iteration_reset = False
    for i in range(1, len(entry_list)):
        temp[i].features = extract_features(entry_list, build_feature_set(features_template, look_back_count), i)
        if next_iteration_reset:
            look_back_count = 1
            next_iteration_reset = False
        else:
            look_back_count += 1
            if constrained is not None:
                if look_back_count > constrained:
                    look_back_count = constrained
        if temp[i].boundary:
            next_iteration_reset = True

    return temp


def preprocess_search_string(string):
    """preprocesses the search string for feature extraction"""
    return string.replace(
        '\\', ' '
    ).strip()


def extract_data_and_make_pairs(users, feature_set):
    """Reads a dictionary of users with entry lists and extract for each users the features and return them as pairs"""
    pairs = []
    print("Extracting features...")
    for user_key in users.keys():
        user = users[user_key]
        user.sort()
        user_with_features = extract_features(user, feature_set)
        pairs = pairs + make_pairs(user_with_features)
    print("Extract label...")
    label_vector = labels_for_pairs(pairs)
    print("Extract data matrix...")
    data_mat = generate_data_matrix(pairs)

    return pairs, data_mat, label_vector


def extract_data_dynamic_and_make_pairs(users, feature_set_template, constrained=None):
    """Reads a dictionary of users with entry lists and extract for each users the features with dynamic look back and return them as pairs"""
    pairs = []
    print("Extracting features...")
    for user_key in users.keys():
        user = users[user_key]
        user.sort()
        user_with_features = extract_features_dynamic(user, feature_set_template, constrained)
        pairs = pairs + make_pairs(user_with_features)
    print("Extract label...")
    label_vector = labels_for_pairs(pairs)
    print("Extract data matrix...")
    data_mat = generate_data_matrix(pairs, dynamic=True)

    return pairs, data_mat, label_vector