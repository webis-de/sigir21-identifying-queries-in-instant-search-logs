import numpy as np


def make_pairs(entry_list):
    """Generates (n-1) pairs out of a list of n log entries"""
    pairs = []
    for i in range(len(entry_list) - 1):
        pairs.append((entry_list[i], entry_list[i + 1]))
    return pairs


def labels_for_pairs(pairs):
    """Extract the (true) label for a list of pairs"""
    labels = np.zeros(len(pairs))
    for i in range(len(pairs)):
        if pairs[i][0].boundary:
            labels[i] = 1
    return labels


def generate_data_matrix(pairs, dynamic=False):
    """Generates a data matrix containing all features of the given pairs"""
    data_matrix = None
    for pair in pairs:
        if data_matrix is not None:
            if dynamic:
                matrix_rows = data_matrix.shape[0]
                feature_rows = pair[1].features.shape[0]
                if matrix_rows > feature_rows:
                    difference = matrix_rows - feature_rows
                    enlarge = np.zeros((difference, 1))
                    pair[1].features = np.row_stack((pair[1].features.reshape((-1, 1)), enlarge))
                elif matrix_rows < feature_rows:
                    difference = feature_rows - matrix_rows
                    enlarge = np.zeros((difference, data_matrix.shape[1]))
                    data_matrix = np.row_stack((data_matrix, enlarge))
                data_matrix = np.column_stack((data_matrix, pair[1].features))
            else:
                data_matrix = np.column_stack((data_matrix, pair[1].features))
        else:
            data_matrix = pair[1].features
            data_matrix = data_matrix.reshape((-1, 1))
    return data_matrix


def save_pairs(pairs, filename):
    """Save pairs to a file"""
    with open(filename, "w") as file:
        for pair in pairs:
            file.write("(" + str(pair[0]) + "," + str(pair[1]) + ")\n")
