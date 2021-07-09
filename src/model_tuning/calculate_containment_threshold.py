import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))


from data.util import *
from models.metrics import evaluate_model
from rule_based.pipeline import execute_rule_pipeline
from rule_based.rulesets import *
from rule_based.parameters import *
from features.featuresets import template_time
from features.featureextraction import build_feature_set, extract_data_and_make_pairs

#parameters

FILE = "webis-NIL-21.ldjson"
TEST_SIZE = 0.1
CROSS_VAL_SIZE = 3

train_pairs_cv, val_pairs_cv, test_pairs_cv, train_users, val_users, test_users = perform_preprocessing_and_get_pairs(FILE, TEST_SIZE, 2429, True, CROSS_VAL_SIZE)

predicted_label, true_label = execute_rule_pipeline(train_pairs_cv[0], time_gap)


train_pairs, data_mat, label_vector_train = extract_data_and_make_pairs(train_users[0], build_feature_set(template_time, look_back_count=1))

print(data_mat.shape)

prefix_suffix_pair_indices  = []
prefix_suffix_pairs = []
for pair in np.nonzero(predicted_label == -1)[0]:
    if containment_step(train_pairs[pair]) == 0:
        prefix_suffix_pairs.append(train_pairs[pair])
        prefix_suffix_pair_indices.append(pair)

print(f"Number of pairs: {len(prefix_suffix_pairs)}/{np.count_nonzero(predicted_label == -1)}")

data_mat = data_mat[:, np.array(prefix_suffix_pair_indices)].reshape(-1)
label_vector_train = label_vector_train[np.array(prefix_suffix_pair_indices)]
print(data_mat.shape)


k = 0
lower = 0
while True:
    lower_delta_time = 0

    j = 0
    l = 0
    for entry in np.argsort(data_mat):
        j += 1
        if label_vector_train[entry] == 1:
            if k > l:
                l += 1
                continue
            lower_delta_time = data_mat[entry]
            break
    print(f"Lower delta time {lower_delta_time}")
    print(f"Split number {k + 1} at position {j}")
    print(f"Pair that does not fit:")
    print(prefix_suffix_pairs[np.argsort(data_mat)[j - 1]])
    lower = lower_delta_time

    next_prev = input("(n)ext or (p)rev or (c)onfirm or (r)un:")
    if next_prev == "n":
        k += 1
    elif next_prev == "p":
        k -= 1
        if k < 0:
            k = 0
    elif next_prev == "c":
        print(f"Lower delta time {lower}")
        LOWER_TIME_PRE_SUF_THRESHOLD = lower
    elif next_prev == "r":
        break
    else:
        investigate_upper_or_lower = input("(u)pper or (l)ower:")
        k = 0


predicted_label_val, true_label_val = execute_rule_pipeline(val_pairs_cv[0], containment)
print(f"Testing parameter and lower: {LOWER_TIME_PRE_SUF_THRESHOLD} on {true_label_val.shape[0]} pairs")
evaluate_model(true_label_val[predicted_label_val != -1], predicted_label_val[predicted_label_val != -1])
false_negatives = np.nonzero(np.logical_and(predicted_label_val == 0, true_label_val == 1))
predicted_label_val[predicted_label_val == -1] = 0
evaluate_model(true_label_val, predicted_label_val)

print(f"False negatives:")
for entry in false_negatives[0].tolist():
    print(entry)
    print(val_pairs_cv[0][entry])
