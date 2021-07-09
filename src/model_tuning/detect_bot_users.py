import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))


from datetime import timedelta

from data.util import *
import numpy as np
import matplotlib.pyplot as plt

#load training data
FILE = "webis-NIL-21.ldjson"
TEST_SIZE = 0.1
CROSS_VAL_SIZE = 10

print("Importing data...")
data = import_data(FILE)



print("Removing non-ASCII")

print("Performing train/testsplit....")
users = list(data.keys())
user_train, user_test = train_test_split(users, test_size=TEST_SIZE, random_state=2429)
train_dict = get_sub_dict(user_train, data)
test_dict = get_sub_dict(user_test, data)

lower_threshold = timedelta(seconds=1)
larger_threshold = timedelta(minutes=15)

def count_activities_in_timespan(user, time_threshold):
    user.sort()
    best = []
    current = []
    prev_entry = None
    for entry in user:
        if prev_entry is None:
            current.append(0)
            best = current
            prev_entry = entry
            continue

        delta_time = abs((prev_entry.timestamp - entry.timestamp).total_seconds())
        if sum(current) + delta_time <= time_threshold.total_seconds():
            current.append(delta_time)
        else:
            while sum(current) + delta_time > time_threshold.total_seconds() and len(current) >= 1:
                current.pop(0)
            if len(current) >= 1:
                current.append(delta_time)
            else:
                current.append(0)
        prev_entry = entry

        if len(current) > len(best):
            best = current
    return len(best)

def count_activities_with_timespan(user, time_threshold):
    user.sort()
    prev_entry = None
    count = 0
    for entry in user:
        if prev_entry is None:
            prev_entry = entry
            continue
        delta_time = abs((prev_entry.timestamp - entry.timestamp).total_seconds())
        if delta_time <= time_threshold.total_seconds():
            count += 1
        prev_entry = entry
    return count

def average_length(user):
    string_length = 0
    for entry in user:
        string_length += len(entry.interaction)
    return string_length / len(user)


def median_length(user):
    string_length = []
    for entry in user:
        string_length.append(len(entry.interaction))
    return np.median(np.array(string_length))

#first step number of users with at least one pair with time delta less then the lower threshold
print("Splitting in pairs and collapsing user...")

collapsed_users = data

pairs_below_threshold = np.zeros((len(collapsed_users),))
pairs_below_large_threshold = np.zeros((len(collapsed_users),))
average_entry_length = np.zeros((len(collapsed_users),))
pair_count_with_threshold = np.zeros((len(collapsed_users),))
i = 0
for user_key in collapsed_users.keys():
    user = collapsed_users[user_key]
    pairs_below_threshold[i] = count_activities_in_timespan(user[0], lower_threshold)
    pairs_below_large_threshold[i] = count_activities_in_timespan(user[0], larger_threshold)
    average_entry_length[i] = median_length(user[0])
    pair_count_with_threshold[i] = count_activities_with_timespan(user[0], timedelta(milliseconds=5))
    i += 1

print(f"Total number of users: {len(collapsed_users)}")
print(f"Users below threshold: {np.count_nonzero(pairs_below_threshold >= 1)}")

plt.figure()
plt.hist(pairs_below_threshold, bins=100)
plt.ylabel("Number of Users")
plt.xlabel("Number of logentries in certain timespan")
plt.show()

plt.figure()
plt.hist(pairs_below_large_threshold, bins=250)
plt.ylabel("Number of Users")
plt.xlabel("Number of logentries in certain timespan (larger)")
plt.show()


plt.figure()
plt.hist(average_entry_length)
plt.ylabel("Number of Users")
plt.xlabel("Average length of logentries")
plt.show()

plt.figure()
plt.hist(pair_count_with_threshold)
plt.ylabel("Number of Users")
plt.xlabel("Pairs below 5ms")
plt.show()
print(f"Users with short timespan: {np.count_nonzero(pair_count_with_threshold > 4)}")
print(f"IPs of this users:")
temp = np.nonzero(pair_count_with_threshold > 4)
ips = []
for user in temp[0].tolist():
    ips.append(list(collapsed_users.keys())[user])
print(ips)

#print users that have exactly n pairs below threshold
n = int(input("Entries above in one second="))
user_with_exactly_n_pairs = np.nonzero(pairs_below_threshold >= n)
ips = []
for user in user_with_exactly_n_pairs[0].tolist():
    ips.append(list(collapsed_users.keys())[user])
print(ips)

k = int(input("Entries above in 30 minutes="))
user_with_exactly_k_pairs = np.nonzero(pairs_below_large_threshold >= k)
ips = []
for user in user_with_exactly_k_pairs[0].tolist():
    ips.append(list(collapsed_users.keys())[user])
print(ips)


#print users that have average length
m = int(input("Average entry length="))
users_with_excactly_m_average = np.nonzero(average_entry_length >= m)
ips = []
for user in users_with_excactly_m_average[0].tolist():
    ips.append(list(collapsed_users.keys())[user])
print(ips)
print(average_entry_length.max())
