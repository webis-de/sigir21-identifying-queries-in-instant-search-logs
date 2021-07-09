import argparse
import pickle
import time

from sklearn.linear_model import LogisticRegression
import numpy as np

from data import save_pairs
from data.util import perform_preprocessing_and_get_pairs
from features.featuresets import feature_template
from models.metrics import evaluate_model
from models.training import model_behind_rules_cv_parallel, train_model_behind_rules, run_model_behind_rules
from rule_based.pipeline import execute_rule_pipeline
from rule_based.rulesets import time_gap, time_gap_final, containment, containment_final, \
    lexical_dissimilarity, lexical_dissimilarity_final, hagen_withoug_semantic_for_lgr, cetendil, kim, \
    hagen_without_semantic, lexical_similarity, lexical_similarity_final

parser = argparse.ArgumentParser(description="Run and validate models on given dataset")
parser.add_argument("dataset",
                    help="Path to dataset")
parser.add_argument("-m", "--model",
                    choices=["lgr", "none"],
                    default="lgr",
                    help="Which model should be used? lgr for logistic regression and none means only rules")
parser.add_argument("--modelpath",
                    default="trained_logistic_regression.pyc",
                    help="Path to pre-trained model")
parser.add_argument("-r", "--rules",
                    choices=["none", "time_gap", "containment", "lexical_similarity", "lexical_dissimilarity",
                             "old_pipeline", "cetendil", "kim", "hagen"],
                    default="lexical_dissimilarity",
                    help="Which rules should be used? None, time difference, time and string containment or time, containment and lexical similarity?")
parser.add_argument("--seed",
                    type=int,
                    default=2429,
                    help="Random seed for train/test split")
parser.add_argument("--dynamic",
                    type=bool,
                    default=False,
                    help="Dynamic lookback enabled")
parser.add_argument("--verbosity",
                    type=int,
                    default=0,
                    help="Level of verbosity")
parser.add_argument("--lookback",
                    type=int,
                    default=3,
                    help="Determins how many log entries our model looks back")
parser.add_argument("--testsize",
                    type=float, default=0.1, help="Which fraction of the dataset should be used for test")
parser.add_argument("--cv", type=int, default=10, help="Determins what level of cross validation (on users) we use")
parser.add_argument("--savefp", type=bool, default=False, help="Save false positives to file.")
parser.add_argument("--savetp", type=bool, default=False, help="Save true positives to file.")
parser.add_argument("--savefn", type=bool, default=False, help="Save false negatives to file.")
parser.add_argument("--test", type=bool, default=False, help="Train on train set and test on test set.")
parser.add_argument("--finalrules", type=bool, default=False, help="Finish after last rule. Uncertain decisions are set opposit to the decision of the last rule.")
parser.add_argument("--train", type=bool, default=False, help="Train and save a model for later use")

args = parser.parse_args()


train_pairs, val_pairs, test_pairs, train_users, val_users, test_users = perform_preprocessing_and_get_pairs(args.dataset,
                                                                                                             test_size=args.testsize,
                                                                                                             random_state_train_testsplit=args.seed,
                                                                                                             return_users=True,
                                                                                                             cross_val_size=args.cv)

logistic_regression = LogisticRegression(solver="liblinear", max_iter=1_000_000)

if args.model == "lgr":
    model = logistic_regression
elif args.model == "none":
    model = None

if args.rules == "none":
    ruleset = []
elif args.rules == "time_gap":
    if model is not None or not args.finalrules:
        ruleset = time_gap
    else:
        ruleset = time_gap_final
elif args.rules == "containment":
    if model is not None or not args.finalrules:
        ruleset = containment
    else:
        ruleset = containment_final
elif args.rules == "lexical_similarity":
    if model is not None or not args.finalrules:
        ruleset = lexical_similarity
    else:
        ruleset = lexical_similarity_final
elif args.rules == "lexical_dissimilarity":
    if model is not None or not args.finalrules:
        ruleset = lexical_dissimilarity
    else:
        ruleset = lexical_dissimilarity_final
elif args.rules == "old_pipeline":
    ruleset = hagen_withoug_semantic_for_lgr
elif args.rules == "hagen":
    ruleset = hagen_without_semantic
elif args.rules == "cetendil":
    ruleset = cetendil
elif args.rules == "kim":
    ruleset = kim

if model is not None:
    if not args.test and not args.train:
        model_results = model_behind_rules_cv_parallel(model=model,
                                                       rule_set=ruleset,
                                                       data=args.dataset,
                                                       test_size=args.testsize,
                                                       test_split_seed=args.seed,
                                                       cv_size=args.cv,
                                                       verbose=args.verbosity,
                                                       dynamic=args.dynamic,
                                                       static_lookback=args.lookback,
                                                       return_tp=args.savetp)

        for i in range(args.cv):
            if args.savefp:
                save_pairs(model_results["fp_pairs"][i], "fp_pairs_run" + str(i) + ".txt")
            if args.savefn:
                save_pairs(model_results["fn_pairs"][i], "fn_pairs_run" + str(i) + ".txt")
            if args.savetp:
                save_pairs(model_results["tp_pairs"][i], "tp_pairs_run" + str(i) + ".txt")

        print(f"Results: ACC = {model_results['acc'].mean():.2f} (+-{model_results['acc'].std():.2f})"
              f" PRE = {model_results['pre'].mean():.2f} (+-{model_results['pre'].std():.2f})"
              f" REC = {model_results['rec'].mean():.2f} (+-{model_results['rec'].std():.2f})"
              f" F_2 = {model_results['f2'].mean():.2f} (+-{model_results['f2'].std():.2f})"
              f" TP = {model_results['tp'].mean():.2f} (+-{model_results['tp'].std():.2f})"
              f" TN = {model_results['tn'].mean():.2f} (+-{model_results['tn'].std():.2f})"
              f" FP = {model_results['fp'].mean():.2f} (+-{model_results['fp'].std():.2f})"
              f" FN = {model_results['fn'].mean():.2f} (+-{model_results['fn'].std():.2f})")
    elif args.test:
        train_pairs, test_pairs, train_users, test_users = perform_preprocessing_and_get_pairs(args.dataset,
                                                                                               args.testsize,
                                                                                               args.seed,
                                                                                               return_users=True)

        print("Loading pre trained model...")
        print(args.modelpath)
        model = pickle.load(open(args.modelpath, "rb"), fix_imports=True)

        print("Testing model...")
        start_time = time.time()
        results = run_model_behind_rules(model,
                                         test_users,
                                         ruleset,
                                         feature_template,
                                         dynamic=args.dynamic,
                                         static_lookback=args.lookback)
        end_time = time.time()
        test_entries = 0
        for user in test_users:
            test_entries += len(test_users[user])
        print(f"Needed time: {(end_time - start_time) * 1000:.4f} ms")
        print(f"Per pair: {(end_time - start_time) * 1000 / test_entries:.4f} ms")
        evaluate_model(results["true_label"], results["predicted_label"])
    elif args.train:
        train_pairs, test_pairs, train_users, test_users = perform_preprocessing_and_get_pairs(args.dataset,
                                                                                               args.testsize,
                                                                                               args.seed,
                                                                                               return_users=True)
        users = train_users
        users.update(test_users)
        model, feature_selection = train_model_behind_rules(model,
                                                            users,
                                                            ruleset,
                                                            feature_template,
                                                            dynamic=args.dynamic,
                                                            static_lookback=args.lookback)

        print("Saving pre trained model...")
        with open("trained_model.pyc", "wb") as outputfile:
            pickle.dump(model, outputfile)

else:
    train_pairs, test_pairs = perform_preprocessing_and_get_pairs(args.dataset,
                                                                  args.testsize,
                                                                  args.seed)
    fp_pairs = []
    tp_pairs = []
    fn_pairs = []
    start_time = time.time()
    total_pairs = 0
    if args.test:
        predicted_label, true_label = execute_rule_pipeline(test_pairs, ruleset)
        total_pairs = len(test_pairs)

        for pair in np.argwhere(np.logical_and(predicted_label == 1, true_label == 0)).tolist():
            fp_pairs.append(test_pairs[pair[0]])

        for pair in np.argwhere(np.logical_and(predicted_label == 0, true_label == 1)).tolist():
            fn_pairs.append(test_pairs[pair[0]])

        for pair in np.argwhere(np.logical_and(predicted_label == 1, true_label == 1)).tolist():
            tp_pairs.append(test_pairs[pair[0]])

    else:
        predicted_label, true_label = execute_rule_pipeline(train_pairs, ruleset)
        total_pairs = len(train_pairs)

        for pair in np.argwhere(np.logical_and(predicted_label == 1, true_label == 0)).tolist():
            fp_pairs.append(train_pairs[pair[0]])

        for pair in np.argwhere(np.logical_and(predicted_label == 0, true_label == 1)).tolist():
            fn_pairs.append(train_pairs[pair[0]])

        for pair in np.argwhere(np.logical_and(predicted_label == 1, true_label == 1)).tolist():
            tp_pairs.append(train_pairs[pair[0]])

    print(f"Total pairs: {total_pairs}")
    print(f"Predicted pairs: {np.count_nonzero(predicted_label != -1)}")
    print(f"Percent decided: {np.count_nonzero(predicted_label != -1) * 100./total_pairs:.2f}%")
    print(f"Metrics for this pairs:")
    evaluate_model(true_label[np.nonzero(predicted_label != -1)], predicted_label[predicted_label != -1])
    end_time = time.time()
    print(f"Needed time: {(end_time - start_time) * 1000} ms")
    print(f"Time per pair: {(end_time - start_time) * 1000 / total_pairs} ms")
    if args.savefp:
        save_pairs(fp_pairs, "fp_pairs_rules.txt")
    if args.savefn:
        save_pairs(fn_pairs, "fn_pairs_rules.txt")
    if args.savetp:
        save_pairs(tp_pairs, "tp_pairs_rules.txt")
