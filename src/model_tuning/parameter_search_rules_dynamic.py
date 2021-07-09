import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))


from sklearn.linear_model import LogisticRegression

from data.util import *
from models.training import optimize_model
from rule_based.pipeline import execute_rule_pipeline
from rule_based.rulesets import *
from models.metrics import evaluate_model
from features.featuresets import feature_template
from features.featureextraction import extract_data_dynamic_and_make_pairs

#load training data

FILE = "webis-NIL-21.ldjson"
TEST_SIZE = 0.1
CROSS_VAL_SIZE = 10


train_pairs, test_pairs, train_users, test_users = perform_preprocessing_and_get_pairs(FILE, TEST_SIZE, 2429, True)

predicted_label, true_label = execute_rule_pipeline(train_pairs, lexical_dissimilarity)

print(f"Total pairs: {len(train_pairs)}")
print(f"Predicted pairs rule based: {np.count_nonzero(predicted_label != -1)}")
print(f"Metrics for this pairs:")
evaluate_model(true_label[np.nonzero(predicted_label != -1)], predicted_label[predicted_label != -1])

print(f"Overall metrics (rule based; non predicted entries set to 0)")
predicted_label_correct = np.copy(predicted_label)
predicted_label_correct[predicted_label == -1] = 0
evaluate_model(true_label, predicted_label_correct)

logistic_regression_parameter_grid = {"C": [0.01, 0.1, 0.25, 0.5, 1, 1.5, 2, 3, 5]}

logistic_regression = LogisticRegression(solver="liblinear", max_iter=1_000_000, n_jobs = 24)


print("Extracting features...")
train_pairs, data_mat, label_vector = extract_data_dynamic_and_make_pairs(train_users, feature_template)

print(data_mat.shape)

data_mat_train = data_mat[:, np.nonzero(predicted_label == -1)[0]]
print(data_mat_train.shape)

print("Running logistic regression...")
optimize_model(logistic_regression, np.transpose(data_mat_train_selected), label_vector[np.nonzero(predicted_label == -1)[0]], CROSS_VAL_SIZE, "logistic_regression_gridsearch")




