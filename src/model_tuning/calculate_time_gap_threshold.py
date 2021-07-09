import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))


from data.util import *
from features.featuresets import template_time
from features.featureextraction import build_feature_set, extract_data_and_make_pairs

#load training data
FILE = "webis-NIL-21.ldjson"
TEST_SIZE = 0.1
CROSS_VAL_SIZE = 10


train_pairs, test_pairs, train_users, test_users = perform_preprocessing_and_get_pairs(FILE, TEST_SIZE, 2429, True)


train_pairs, data_mat, label_vector = extract_data_and_make_pairs(train_users, build_feature_set(template_time, look_back_count=1))

data_mat = data_mat[0]

print(f"Entrys with no time difference: {np.count_nonzero(data_mat == 0.0)}")
print("Saving those pairs to file: ")
with open("../delta_time_equals_zero.txt", "w") as file:
    for pair in np.nonzero(data_mat == 0.0)[0]:
        file.write("(" + str(train_pairs[pair][0]) + "," + str(train_pairs[pair][1]) + ")")

upper_delta_time = 0

i = 0
for entry in np.flip(np.argsort(data_mat)):
    i += 1
    if label_vector[entry] == 0:
        upper_delta_time = data_mat[entry]
        break

lower_delta_time = 0

j = 0
for entry in np.argsort(data_mat):
    j += 1
    if label_vector[entry] == 1:
        lower_delta_time = data_mat[entry]
        break

print(f"Splits: {np.count_nonzero(label_vector == 1)}")
print(f"Total pairs: {label_vector.shape[0]}")
print(f"Position of first non split from above: {i}")
print(f"Upper delta time {upper_delta_time}")
print(f"Position of first split from below: {j}")
print(f"Lower delta time {lower_delta_time}")
