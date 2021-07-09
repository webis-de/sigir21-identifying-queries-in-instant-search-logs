import multiprocessing
import copy

from sklearn.metrics import accuracy_score, precision_score, recall_score, fbeta_score
from sklearn.model_selection import GridSearchCV

from rule_based.pipeline import execute_rule_pipeline
from .metrics import tp, tn, fp, fn, evaluate_model
from data.util import *
from features.featuresets import feature_template
from features.featureextraction import build_feature_set, extract_features, extract_data_and_make_pairs, \
    extract_data_dynamic_and_make_pairs


def optimize_model(model, X, y, parameters, cross_val_size=10, filename=None, n_jobs=24, verbose=3):
    """Optimize the given model over the given data and save the results if filename is not None"""
    searcher = GridSearchCV(model, parameters, "recall", n_jobs=n_jobs, cv=cross_val_size, verbose=verbose)
    print("Performing GridSearch...")
    searcher.fit(X, y)
    if filename is not None:
        np.save(filename, searcher.cv_results_)
    else:
        return searcher.cv_results_


def model_behind_rules(model, rule_set, train_users, val_users, dynamic=False, static_lookback=3,
                       verbose=0, return_tp=False):
    """Trains a model behind the rules first and validates it after this."""
    if model is not None:
        model, feature_selection, stats, rules_label_train, true_label_train = train_model_behind_rules(model,
                                                                                                        train_users,
                                                                                                        rule_set,
                                                                                                        feature_template,
                                                                                                        dynamic,
                                                                                                        static_lookback,
                                                                                                        verbose,
                                                                                                        return_train_label_and_stats=True)

    results = run_model_behind_rules(model, val_users, rule_set, feature_template, dynamic, static_lookback,
                                     return_tp, verbose, -1)
    if model is not None:
        results["stats"] = {**results["stats"], **stats}
        results["rules_label_train"] = rules_label_train
        results["true_label_train"] = true_label_train

    return results


def model_behind_rules_cv_parallel(model, rule_set, data, test_size, test_split_seed, cv_size=10, verbose=0, dynamic=False,
                                   static_lookback=3, return_tp=False):
    """Trains multiple models behind the rules first and validates them. Runs parallel."""
    train_pairs, val_pairs, test_pairs, train_users, val_users, test_users = perform_preprocessing_and_get_pairs(data,
                                                                                                                 test_size,
                                                                                                                 test_split_seed,
                                                                                                                 True,
                                                                                                                 cv_size)

    processes = []
    output = multiprocessing.Queue()

    def run(train_users, val_users, output_queue, i=-1):
        model_results = model_behind_rules(model, rule_set, train_users, val_users, dynamic, static_lookback,
                                           verbose, return_tp)
        if return_tp:
            output_queue.put((i, model_results['predicted_label'], model_results['rules_label'],
                                 model_results['true_label'],      model_results['fp_pairs'],
                                 model_results['fn_pairs'],        model_results['stats'],
                                 model_results['tp_pairs']))
        else:
            output_queue.put((i, model_results['predicted_label'], model_results['rules_label'],
                              model_results['true_label'], model_results['fp_pairs'],
                              model_results['fn_pairs'], model_results['stats']))

    for i in range(cv_size):
        if verbose >= 1:
            print(i)
        processes.append(multiprocessing.Process(target=run, args=(copy.deepcopy(train_users[i]),
                                                                   copy.deepcopy(val_users[i]),
                                                                   output,
                                                                   i)))

    if verbose >= 1:
        print("Starting processes...")
    for process in processes:
        process.start()

    results = {"predicted_label_val": [None] * cv_size,
               "rules_label_val": [None] * cv_size,
               "true_label_val": [None] * cv_size,
               "fp_pairs": [None] * cv_size,
               "fn_pairs": [None] * cv_size,
               "tp_pairs": [None] * cv_size,
               "acc": np.zeros((cv_size,)),
               "pre": np.zeros((cv_size,)),
               "rec": np.zeros((cv_size,)),
               "f2": np.zeros((cv_size,)),
               "tp": np.zeros((cv_size,)),
               "fp": np.zeros((cv_size,)),
               "tn": np.zeros((cv_size,)),
               "fn": np.zeros((cv_size,)),
               "stats": [None] * cv_size}

    for process in processes:
        temp = output.get()
        i = temp[0]
        if verbose >= 1:
            print(i)
        results["predicted_label_val"][i] = temp[1]
        results["rules_label_val"][i] = temp[2]
        results["true_label_val"][i] = temp[3]
        results["fp_pairs"][i] = temp[4]
        results["fn_pairs"][i] = temp[5]
        results["stats"][i] = temp[6]
        if return_tp:
            results["tp_pairs"][i] = temp[7]
        results["acc"][i] = accuracy_score(results["true_label_val"][i], results["predicted_label_val"][i])
        results["pre"][i] = precision_score(results["true_label_val"][i], results["predicted_label_val"][i])
        results["rec"][i] = recall_score(results["true_label_val"][i], results["predicted_label_val"][i])
        results["tp"][i] = tp(results["true_label_val"][i], results["predicted_label_val"][i])
        results["fp"][i] = fp(results["true_label_val"][i], results["predicted_label_val"][i])
        results["tn"][i] = tn(results["true_label_val"][i], results["predicted_label_val"][i])
        results["fn"][i] = fn(results["true_label_val"][i], results["predicted_label_val"][i])
        results["f2"][i] = fbeta_score(results["true_label_val"][i], results["predicted_label_val"][i], beta=2)

    return results


def train_model_behind_rules(model, train_users, rule_set, feature_set_template, dynamic=False, static_lookback=3,
                             verbose=0, return_train_label_and_stats=False):
    """ Run all rules and train the model on the remaining, undecided entries.

    :param model: The model which should be trained.
    :param train_users: A dictionary of the users used for training.
    :param rule_set: The rule set which will be executed before.
    :param feature_set_template: Template of features used for training
    :param dynamic: if True the number of entries we are looking back is determined dynamic
    :param static_lookback: number of entries which we are looking back; overrided by dynamic
    :param verbose: level of verbosity
    :param return_train_label_and_stats: if True simple statistics and the labels for the rules/true labels are returned
    :return: the trained model
    """
    train_pairs = []
    stats = {}
    for user in train_users:
        train_pairs += make_pairs(train_users[user])

    if verbose >= 1:
        print("Running rules")

    rules_label_train, true_label_train = execute_rule_pipeline(train_pairs, rule_set)

    stats["pairs_train"] = len(train_pairs)
    stats["decided_by_rules_train"] = np.count_nonzero(rules_label_train != -1)

    if verbose >= 2:
        print(f"Total pairs train: {stats['pairs_train']}")
        print(f"Predicted pairs rule based: {stats['decided_by_rules_train']}")
        print(f"Metrics for this pairs:")
        evaluate_model(true_label_train[rules_label_train != -1],
                       rules_label_train[rules_label_train != -1])

        print(f"Overall metrics (rule based; non predicted entries set to 0)")
        predicted_label_correct = np.copy(rules_label_train)
        predicted_label_correct[rules_label_train == -1] = 0
        evaluate_model(true_label_train, predicted_label_correct)

    if verbose >= 1:
        print("Extracting train features")

    if dynamic:
        train_pairs, data_mat, label_vector = extract_data_dynamic_and_make_pairs(train_users, feature_set_template)
    else:
        train_pairs, data_mat, label_vector = extract_data_and_make_pairs(train_users,
                                                                          build_feature_set(feature_set_template,
                                                                                            look_back_count=static_lookback))

    data_mat_train = data_mat[:, rules_label_train == -1]

    if verbose >= 1:
        print("Fitting model...")
    model.fit(np.transpose(data_mat_train), true_label_train[np.nonzero(rules_label_train == -1)])

    if return_train_label_and_stats:
        return model, stats, rules_label_train, true_label_train
    else:
        return model


def run_model_behind_rules(model, users, rule_set, feature_set_template, dynamic=False, static_lookback=3,
                           return_tp=False, verbose=0):
    """Run the entries through the rules. Still undecided pairs are decided by the model.

    :param model: The model which should be used.
    :param users: A dictionary of the users on which the model should be used.
    :param rule_set: The rule set which will be executed before.
    :param feature_set_template: Template of features used for the model
    :param dynamic: if True the number of entries we are looking back is determined dynamic
    :param static_lookback: number of entries which we are looking back; overrided by dynamic
    :param return_tp: returns the correct splits additionally
    :param verbose: level of verbosity
    :return: a dictionary containing a list of false positive, false negative, statistics and the label vector for the rules, the prediction of rules and model and the true label
    """

    stats = {}
    fp_pairs = []
    fn_pairs = []
    tp_pairs = []

    if dynamic:
        size = model.coef_.shape[1]
        val_pairs = []
        rules_label_val = None
        true_label_val = None
        predicted_label_val = None
        for user_key in users:
            user = users[user_key]
            user.sort()
            user_pairs = []

            user_pairs_temp = make_pairs(copy.deepcopy(user))
            rules_decision, true_decision = execute_rule_pipeline(user_pairs_temp, rule_set)
            known_splits = np.nonzero(rules_decision == 1)[0].tolist()

            rules_label_user = np.zeros((len(user) - 1,))
            true_label_user = np.zeros((len(user) - 1,))
            pred_label_user = np.zeros((len(user) - 1,))

            look_back_count = 1


            user[0].features = extract_features(user, build_feature_set(feature_set_template, look_back_count), 1)

            difference_back = size - user[0].features.shape[0]
            if difference_back < 0:
                user[0].features = user[0].features[:size]
            else:
                zeros = np.zeros((1, difference_back))[0]
                user[0].features = np.hstack((user[0].features, zeros))

            for i in range(1, len(user)):

                user[i].features = extract_features(user, build_feature_set(feature_set_template, look_back_count), i)

                difference_back = size - user[i].features.shape[0]
                if difference_back < 0:
                    user[i].features = user[i].features[:size]
                else:
                    zeros = np.zeros((1, difference_back))[0]
                    user[i].features = np.hstack((user[i].features, zeros))

                new_pair = (user[i - 1], user[i])
                user_pairs.append(new_pair)

                rule_prediction, split = execute_rule_pipeline([new_pair], rule_set)
                rules_label_user[i - 1] = rule_prediction[0]
                true_label_user[i - 1] = split[0]

                if rule_prediction[0] == -1 and model is not None:
                    model_prediction = model.predict(new_pair[1].features.reshape(1, -1))
                    pred_label_user[i - 1] = model_prediction[0]
                    new_pair[0].boundary_prediction = bool(model_prediction[0])
                else:
                    pred_label_user[i - 1] = rule_prediction[0]
                    new_pair[0].boundary_prediction = bool(rule_prediction[0])

                if pred_label_user[i - 1] == 1:
                    look_back_count = 1
                else:
                    look_back_count += 1

                if pred_label_user[i - 1] == 0 and true_label_user[i - 1] == 1:
                    fn_pairs.append(new_pair)
                elif pred_label_user[i - 1] == 1 and true_label_user[i - 1] == 0:
                    fp_pairs.append(new_pair)
                elif pred_label_user[i - 1] == 1 and true_label_user[i - 1] == 1:
                    tp_pairs.append(new_pair)

            if rules_label_val is None:
                rules_label_val = rules_label_user
            else:
                rules_label_val = np.hstack((rules_label_val, rules_label_user))

            if predicted_label_val is None:
                predicted_label_val = pred_label_user
            else:
                predicted_label_val = np.hstack((predicted_label_val, pred_label_user))

            if true_label_val is None:
                true_label_val = true_label_user
            else:
                true_label_val = np.hstack((true_label_val, true_label_user))

            val_pairs += user_pairs

        stats["pairs_val"] = len(val_pairs)
        stats["splits_val"] = np.count_nonzero(true_label_val == 1)
        stats["decided_by_rules_val"] = np.count_nonzero(rules_label_val != -1)
        if verbose >= 2:
            print(f"Validation set contains {stats['pairs_val']} pairs")
            print(f"Splits in this set {stats['splits_val']}")

            print(f"{stats['decided_by_rules_val']} pairs were decided by rule based")
            print(f"Metrics: ")
            print(f"Decided by rule:")
            evaluate_model(true_label_val[rules_label_val != -1],
                           rules_label_val[rules_label_val != -1])
            pred_label_corrected = rules_label_val
            pred_label_corrected[pred_label_corrected == -1] = 0
            print(f"Decided by rule (decisions with no confidence set to 0):")
            evaluate_model(true_label_val, pred_label_corrected)

    else: # static look back
        if verbose >= 1:
            print("Extracting validation features")
        val_pairs, data_mat_val, true_label_val = extract_data_and_make_pairs(users, build_feature_set(feature_set_template,
                                                                                                       look_back_count=static_lookback))

        stats["pairs_val"] = len(val_pairs)
        stats["splits_val"] = np.count_nonzero(true_label_val == 1)

        rules_label_val, true_label_val = execute_rule_pipeline(val_pairs, rule_set)
        stats["decided_by_rules_val"] = np.count_nonzero(rules_label_val != -1)

        if verbose >= 2:
            print(f"Validation set contains {stats['pairs_val']} pairs")
            print(f"Splits in this set {stats['splits_val']}")

            print(f"{stats['decided_by_rules_val']} pairs were decided by rule based")
            print(f"Metrics: ")
            print(f"Decided by rule:")
            evaluate_model(true_label_val[rules_label_val != -1],
                           rules_label_val[rules_label_val != -1])
            pred_label_corrected = rules_label_val
            pred_label_corrected[pred_label_corrected == -1] = 0
            print(f"Decided by rule (decisions with no confidence set to 0):")
            evaluate_model(true_label_val, pred_label_corrected)

        predicted_label_val = np.zeros((len(val_pairs),))
        i = 0
        for pair in val_pairs:
            pred, true = execute_rule_pipeline([pair], rule_set)
            if pred[0] != -1:
                predicted_label_val[i] = pred[0]
            else:
                features_pair = pair[1].features.reshape((-1, 1))
                predicted_label_val[i] = model.predict(np.transpose(features_pair))[0]
                if predicted_label_val[i] == 0 and true[0] == 1:
                    fn_pairs.append(pair)
                elif predicted_label_val[i] == 1 and true[0] == 0:
                    fp_pairs.append(pair)
            pair[0].boundary_prediction = bool(predicted_label_val[i])
            i += 1

    if return_tp:
        return {"fp_pairs": fp_pairs,
                "fn_pairs": fn_pairs,
                "tp_pairs": tp_pairs,
                "stats": stats,
                "rules_label": rules_label_val,
                "true_label": true_label_val,
                "predicted_label": predicted_label_val}
    else:
        return {"fp_pairs": fp_pairs,
                "fn_pairs": fn_pairs,
                "stats": stats,
                "rules_label": rules_label_val,
                "true_label": true_label_val,
                "predicted_label": predicted_label_val}
