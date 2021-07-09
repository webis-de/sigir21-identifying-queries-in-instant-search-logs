from json.decoder import JSONDecodeError

from sklearn.model_selection import train_test_split, KFold

from stats.stats import count_log_entries_in_dict
from .cleaning import remove_non_ascii_strings, remove_users_with_single_entry
from .logentry import LogEntry
from features import count_previous_occurrence
from .pairing import *


def get_sub_dict(keys, dictionary):
    """Returns a sub dictionary with the specified keys"""
    sub_dictionary = {}
    for key in keys:
        sub_dictionary[key] = dictionary[key]
    return sub_dictionary


def get_sub_dict_by_index(keys, dict):
    """Return sub dict by key list index"""
    temp = {}
    for key in keys:
        temp[list(dict.keys())[key]] = dict[list(dict.keys())[key]]
    return temp


def import_data(file_name):
    """Imports the log data"""
    data = {}
    with open(file_name) as file:
        line = file.readline()
        while line:
            try:
                entry = LogEntry(line)
                if entry.ip is not None:
                    if entry.ip in data:
                        data[entry.ip].append(entry)
                    else:
                        data[entry.ip] = [entry]
                else:
                    if entry.uid in data:
                        data[entry.uid].append(entry)
                    else:
                        data[entry.uid] = [entry]
            except JSONDecodeError:
                print("Skipping Entry!")
            line = file.readline()
    return data


def perform_preprocessing_and_get_pairs(file, test_size, random_state_train_testsplit, return_users=False, cross_val_size=None):
    """Load a log data set, perform preprocessing, split it in pairs and returns it as train/test set

    :param file: path to data set or data set itself
    :param test_size: fraction of data set used for testing
    :param random_state_train_testsplit:  seed for the randomized test split
    :param return_users: return additionally the users as dictionary not only the pairs
    :param cross_val_size: size of the validation part for cross validation; None for no cross validation
    :return: the pairs (and users) for training, validation and test
    """

    if type(file) == str:
        print("Importing data...")
        data = import_data(file)
    else:
        data = file

    print(f"Total users: {len(data.keys())} entries: {count_log_entries_in_dict(data)}")

    print("Removing non-ASCII")
    data = remove_non_ascii_strings(data, remove_space=True)
    data = remove_users_with_single_entry(data)

    count_previous_occurrence(data)

    print("Performing train/testsplit....")
    users = list(data.keys())
    user_train, user_test = train_test_split(users, test_size=test_size, random_state=random_state_train_testsplit, shuffle=True)
    print(f"Train users: {len(user_train)}, test users: {len(user_test)}")
    train_dict = get_sub_dict(user_train, data)
    print(f"Train entries: {count_log_entries_in_dict(train_dict)}")

    users_test = test_dict = get_sub_dict(user_test, data)
    print(f"Test entries: {count_log_entries_in_dict(test_dict)}")

    test_pairs = []
    for user_key in test_dict.keys():
        test_pairs += make_pairs(test_dict[user_key])

    train_pairs = []
    if cross_val_size is not None:
        val_pairs = []
        users_train = []
        users_validation = []

        kfold = KFold(n_splits=cross_val_size, random_state=random_state_train_testsplit, shuffle=True)
        users_indices = np.array(list(train_dict.keys()))
        for train_index, test_index in kfold.split(users_indices):

            train_dict_local = get_sub_dict(users_indices[train_index], train_dict)
            pairs = []
            for user_key in train_dict_local.keys():
                pairs += make_pairs(train_dict_local[user_key])

            users_train.append(train_dict_local)
            train_pairs.append(pairs)

            val_dict = get_sub_dict(users_indices[test_index], train_dict)
            pairs_val = []
            for user_key in val_dict.keys():
                pairs_val += make_pairs(val_dict[user_key])
            val_pairs.append(pairs_val)
            users_validation.append(val_dict)

    else:
        users_train = train_dict
        for user_key in train_dict.keys():
            train_pairs += make_pairs(train_dict[user_key])

    if return_users:
        if cross_val_size is None:
            return train_pairs, test_pairs, users_train, users_test
        else:
            return train_pairs, val_pairs, test_pairs, users_train, users_validation, users_test
    else:
        if cross_val_size is None:
            return train_pairs, test_pairs
        else:
            return train_pairs, val_pairs, test_pairs
