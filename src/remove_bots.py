from datetime import timedelta

from data.util import import_data


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

def average_length(user):
    string_length = 0
    for entry in user:
        string_length += len(entry.interaction)
    return string_length / len(user)

FILE="webis-NIL-21.ldjson"
print("Importing data...")
data = import_data(FILE)

print(f"Users before: {len(data.keys())}")


data_cleaned_1 = {}
# (1) more than 300 log entries in 15 minutes
for user_key in data.keys():
    if count_activities_in_timespan(data[user_key], timedelta(minutes=15)) <= 300:
        data_cleaned_1[user_key] = data[user_key]

print(f"Users after step (1): {len(data_cleaned_1.keys())}")


data_cleaned_2 = {}
# (2) average string length > 50
for user_key in data_cleaned_1.keys():
    if average_length(data_cleaned_1[user_key]) <= 50:
        data_cleaned_2[user_key] = data_cleaned_1[user_key]

print(f"Users after step (2): {len(data_cleaned_2.keys())}")

data_cleaned_3 = {}
# (3) more than 10 logentries in one second
for user_key in data_cleaned_2.keys():
    if count_activities_in_timespan(data_cleaned_2[user_key]) <= 10:
        data_cleaned_3[user_key] = data_cleaned_2[user_key]

print(f"Users after step (3): {len(data_cleaned_3.keys())}")

print("Saving users to file")
with open(FILE + ".cleaned", "w") as output:
    for user_key in data_cleaned_3.keys():
        for entry in data_cleaned_3[user_key]:
            output.write(entry.create_json())
            output.write("\n")

