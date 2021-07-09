import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))


from data.util import *
from rule_based.pipeline import execute_rule_pipeline
from rule_based.rulesets import *
from features.featuresets import template_similarity_cosine, template_similarity_jaccard_time
from features.featureextraction import build_feature_set, extract_data_and_make_pairs
import matplotlib.pyplot as plt
from models.metrics import evaluate_model
from rule_based.parameters import *

#parameters
n = 3
binning_number = 20

FILE = "/mnt/data/netspeakdata/annotated/annotated-by-hkp-v6/annotated-by-hkp.ldjson"
TEST_SIZE = 0.1
CROSS_VAL_SIZE = 3


train_pairs_cv, val_pairs_cv, test_pairs_cv, train_users, val_users, test_users = perform_preprocessing_and_get_pairs(FILE, TEST_SIZE, 2429, True, CROSS_VAL_SIZE)

predicted_label, true_label = execute_rule_pipeline(train_pairs_cv[1], containment)



template_similarity_cosine[0][1]["n"]=n
template_similarity_jaccard_time[0][1]["n"]=n

train_pairs, data_mat, label_vector = extract_data_and_make_pairs(train_users[1], build_feature_set(template_similarity_jaccard_time, look_back_count=1))

print(data_mat.shape)





train_pairs_use = []
train_pairs_use_indices = []
for entry in np.nonzero(predicted_label == -1)[0].tolist():
    if len(train_pairs[entry][0].interaction) > n and len(train_pairs[entry][1].interaction) > n:
        train_pairs_use.append(train_pairs[entry])
        train_pairs_use_indices.append(entry)
train_pairs = train_pairs_use
print(len(train_pairs))

time_for_pairs = data_mat[1, np.array(train_pairs_use_indices)].reshape(-1)
data_mat = data_mat[0,np.array(train_pairs_use_indices)].reshape(-1)
label_vector = label_vector[np.array(train_pairs_use_indices)]
print(data_mat.shape)

# perform binning
bins = np.linspace(0, 1, num=binning_number + 1)
binned_labels = np.digitize(data_mat, bins)
print(bins)
print(data_mat.shape)
print(binned_labels.shape)
print(label_vector.shape)
print(np.count_nonzero(label_vector))

plt.figure()
plt.hist(data_mat, bins=bins)
plt.xlabel("similarity")
plt.ylabel("number of users")
plt.show()

def draw_labels_out_of_bin(labels, binned_labels, bin, data=None):
    bin_labels = np.nonzero(binned_labels == bin)
    if data is None:
        return labels[bin_labels]
    else:
        return labels[bin_labels], data[bin_labels]

running = True

time_threshold = None
threshold = 0
time = 0
while running:
    draw_bin = int(input("bin="))
    labels_drawn = draw_labels_out_of_bin(label_vector, binned_labels, draw_bin)
    split_count = np.count_nonzero(labels_drawn)
    print(f"Total number of pairs: {data_mat.shape[0]}")
    print(f"Total number of splits: {np.count_nonzero(label_vector)}")
    print(f"Number of pairs in bin {draw_bin} ({bins[draw_bin-1]} <= x < {bins[draw_bin]}): {labels_drawn.shape[0]}")
    print(f"Number of splits in this bin: {np.count_nonzero(labels_drawn)}")
    if split_count != 0:
        print(f"Lowest split time: {np.min(time_for_pairs[np.logical_and(binned_labels == draw_bin, label_vector == 1)])}")
    print(f"Fraction of splits in this bin: {np.count_nonzero(labels_drawn)/(labels_drawn.shape[0]+0.00000001)}")
    if time_threshold is None:
        if split_count != 0:
            print(f"Number of pairs belonging together in this bin: {np.count_nonzero(labels_drawn == 0)} "
                  f"below split threshold: {np.count_nonzero(np.logical_and(labels_drawn == 0, time_for_pairs[binned_labels == draw_bin] < np.min(time_for_pairs[np.logical_and(binned_labels == draw_bin, label_vector == 1)])))}"
                  f" above split threshold: {np.count_nonzero(np.logical_and(labels_drawn == 0, time_for_pairs[binned_labels == draw_bin] > np.min(time_for_pairs[np.logical_and(binned_labels == draw_bin, label_vector == 1)])))}")
        else:
            print(f"Number of pairs belonging together in this bin: {np.count_nonzero(labels_drawn == 0)} ")
    else:
        print(f"Number of pairs belonging together in this bin: {np.count_nonzero(labels_drawn == 0)} "
              f"below split threshold: {np.count_nonzero(np.logical_and(labels_drawn == 0, time_for_pairs[binned_labels == draw_bin] < time_threshold))}"
              f" above split threshold: {np.count_nonzero(np.logical_and(labels_drawn == 0, time_for_pairs[binned_labels == draw_bin] > time_threshold))}")
        print(f"Number of splits below threshold: {np.count_nonzero(np.logical_and(labels_drawn == 1, time_for_pairs[binned_labels == draw_bin] < time_threshold))}"
              f" and above threshold: {np.count_nonzero(np.logical_and(labels_drawn == 1, time_for_pairs[binned_labels == draw_bin] > time_threshold))}")
    print(f"Highest non-split time: {np.max(time_for_pairs[np.logical_and(binned_labels == draw_bin, label_vector == 0)])}")
    print(f"Fraction of pairs belonging together in this bin: {np.count_nonzero(labels_drawn == 0)/(labels_drawn.shape[0]+0.00000001)}")

    split_indices = np.nonzero(np.logical_and(label_vector == 1, binned_labels == draw_bin))
    non_split_indices = np.nonzero(np.logical_and(label_vector == 0, binned_labels == draw_bin))

    split_or_non_split = input("(s)plit or (n)on split or (c)ancel or (h)istogram or custom (t)ime threshold:")
    if split_or_non_split == "s":
        for index in split_indices[0]:
            print(f"Similarity: {data_mat[index]} for pair {train_pairs[index]}")
    elif split_or_non_split == "n":
        for index in non_split_indices[0]:
            print(f"Similarity: {data_mat[index]} for pair {train_pairs[index]}")
    elif split_or_non_split == "c":
        running = False
        threshold = float(input("Threshold = "))
        time = float(input("Time = "))
    elif split_or_non_split == "t":
        time_threshold = float(input("Threshold = "))
    elif split_or_non_split == "h":
        plt.figure()
        plt.ylabel("number of users")
        type = input("similariy (c)oefficient or (s)plit time distribution or (n)on-split time distribution:")
        if type == "c":
            plt.hist(data_mat, bins=bins)
            plt.xlabel("similarity")
            # plt.xticks(bins, bins, rotation="vertical")
        elif type == "s":
            plt.hist(time_for_pairs[np.logical_and(np.logical_and(label_vector == 1, binned_labels == draw_bin), time_for_pairs <= 20)], bins=40)
            plt.xlabel("time difference")
        else:
            plt.hist(time_for_pairs[np.logical_and(np.logical_and(label_vector == 0, binned_labels == draw_bin), time_for_pairs <= 20)], bins=40)
            plt.xlabel("time difference")
        plt.show()


SIMILARITY_THRESHOLD[n] = threshold
SIMILARITY_TIME_THRESHOLD[n] = time
JACCARD_THRESHOLD[n] = threshold
JACCARD_TIME_THRESHOLD[n] = time

predicted_label_val, true_label_val = execute_rule_pipeline(val_pairs_cv[1], lexical_similarity)
print(f"Total number of pairs in validation set: {len(val_pairs_cv[1])} already predict: {np.count_nonzero(predicted_label_val != -1)}")
evaluate_model(true_label_val[predicted_label_val != -1], predicted_label_val[predicted_label_val != -1])
false_negatives = np.nonzero(np.logical_and(predicted_label_val == 0, true_label_val == 1))
predicted_label_val[predicted_label_val == -1] = 0
evaluate_model(true_label_val, predicted_label_val)

print(f"False negatives:")
for entry in false_negatives[0].tolist():
    print(entry)
    print(val_pairs_cv[1][entry])
