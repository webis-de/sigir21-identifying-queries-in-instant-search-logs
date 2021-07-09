from datetime import timedelta

from data.util import import_data
from stats.stats import calculate_average_query_length
from features import preprocess_search_string, time_gap
import numpy as np

logfile = "webis-NIL-21.ldjson"
data = import_data(logfile)

print(f"Number of users: {len(data.keys())}")

entry_counts = []
query_counts = []
peak_length_char = []
peak_length_terms = []
queries_with_operators = 0
query_duration = []
queries_by_user = {}
for user_key in data.keys():
    entry_counts.append(len(data[user_key]))
    queries_by_user[user_key] = []
    query = []
    for entry in data[user_key]:
        query.append(entry)
        if entry.boundary:
            peak_length_char.append(len(query[-1].interaction))
            peak_length_terms.append(len(query[-1].interaction.split()))
            query_duration.append(abs((query[-1].timestamp - query[0].timestamp).total_seconds()))
            unary_operators = ["...", "?", "#"]
            binary_operators = [["{", "}"], ["[", "]"]]
            string1 = preprocess_search_string(query[-1].interaction)
            contains_operator = False
            for operator in unary_operators:
                if operator in string1:
                    contains_operator = True
            for operator in binary_operators:
                if operator[0] in string1 and operator[1] in string1:
                    contains_operator = True
            if contains_operator:
                queries_with_operators += 1

            queries_by_user[user_key].append(query)
            query = []
    query_counts.append(len(queries_by_user[user_key]))

#interactions per time frame
interactions_per_time_frame = []
queries_per_time_frame = []
for user_key in data.keys():
    interactions_in_frame = 1
    queries_in_frame = 1
    for i in range(1, len(data[user_key])):
        if time_gap(data[user_key][i - 1].timestamp, data[user_key][i].timestamp) > timedelta(minutes=5).total_seconds():
            interactions_per_time_frame.append(interactions_in_frame)
            queries_per_time_frame.append(queries_in_frame)
            interactions_in_frame = 1
            queries_in_frame = 1
        else:
            if entry.boundary:
                queries_in_frame += 1
            interactions_in_frame += 1


print(f"Total entry count: {sum(entry_counts)}")
print(f"Mean entry counts (per user): {sum(entry_counts)/len(entry_counts):.2f}")
print(f"Most entries single user: {max(entry_counts)}, Least entries single user: {min(entry_counts)}")
print(f"Total number of queries: {sum(query_counts)}")
print(f"Most entries single user: {max(query_counts)}, Least entries single user: {min(query_counts)}")
print(f"Queries with operator: {queries_with_operators}")
print(f"Mean query counts (per user): {sum(query_counts)/len(query_counts):.2f}")
print(f"Average query length: {calculate_average_query_length(data):.2f}")
print(f"Average peak query length (characters): {sum(peak_length_char)/len(peak_length_char):.2f}")
print(f"Average peak query length (terms): {sum(peak_length_terms)/len(peak_length_terms):.2f}")
print(f"Median query duration (seconds): {np.median(np.array(query_duration)):.2f}")
print(f"Interactions per physical session: {sum(interactions_per_time_frame)/len(interactions_per_time_frame):.2f}")
print(f"Queries per physical session: {sum(queries_per_time_frame)/len(queries_per_time_frame):.2f}")