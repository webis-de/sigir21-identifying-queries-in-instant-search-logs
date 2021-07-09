from datetime import timedelta
import matplotlib.pyplot as plt
import numpy as np

from data.util import import_data

# path to splitted log files
from stats.stats import operator_use_count, times_occuring, time_between_first_and_last_query, count_see_saw

logfile_instant = "/mnt/data/netspeakdata/instant-splitted.ldjson"
# path to json file containing only the queries
queryfile_instant = "/mnt/data/netspeakdata/parsed_netspeak3_queries_only.log.ldjson"

print("Loading instant log")
instant_queries = import_data(queryfile_instant)

queries_date_start = instant_queries[list(instant_queries.keys())[0]][0].timestamp
queries_date_end = instant_queries[list(instant_queries.keys())[0]][0].timestamp
query_count_per_user = []
queries_with_operator_per_user = []
for user in instant_queries:
    user_sort = sorted(instant_queries[user])
    if queries_date_start > user_sort[0].timestamp:
        queries_date_start = user_sort[0].timestamp
    if queries_date_end < user_sort[-1].timestamp:
        queries_date_end = user_sort[-1].timestamp
    query_count_per_user.append(len(instant_queries[user]))
    queries_with_operator_per_user.append(operator_use_count(instant_queries[user]))
query_count_per_user = np.array(query_count_per_user)
queries_with_operator_per_user = np.array(queries_with_operator_per_user)

print(f"Earliest query: {queries_date_start}, last query:{queries_date_end}")
print(f"No. of users using operators: {np.count_nonzero(queries_with_operator_per_user)}")
print(f"No. of queries: {query_count_per_user.sum()}")
print(f"No. of queries with operators: {queries_with_operator_per_user.sum()}")

active_days = []
time_between_first_and_last = []
for user in instant_queries:
    active_days.append(times_occuring(instant_queries[user]))
    time_between_first_and_last.append(time_between_first_and_last_query(instant_queries[user]))
active_days = np.array(active_days)
time_between_first_and_last = np.array(time_between_first_and_last)


print(f"Users active more than 1 day:{np.count_nonzero(active_days >= 2)} "
      f"({np.count_nonzero(active_days >= 2)/active_days.shape[0]*100:.1f}%)")
print(f"Users still active after one year:{np.count_nonzero(time_between_first_and_last >= timedelta(days=365).total_seconds())}")

triples = {}
triple_count = []
qualifying_triple_count = []
for user in instant_queries:
    qualifying_triple_count.append(count_see_saw(instant_queries[user]))
qualifying_triple_count = np.array(qualifying_triple_count)


print(f"See-Saw-Pattern times: {qualifying_triple_count.sum()}")
print(f"Users which showed see-saw-pattern at least once: {np.count_nonzero(qualifying_triple_count >= 1)}")
print(f"see-saw of users occured at least twice: {qualifying_triple_count[active_days >= 2].sum()}")
print(f"Users that showed see-saw at least once and occured at least twice: "
      f"{np.count_nonzero(np.logical_and(qualifying_triple_count >= 1, active_days >= 2))}")
print(f"See-Saw of users still active after on year: {qualifying_triple_count[time_between_first_and_last >= timedelta(days=365).total_seconds()].sum()}")
print(f"Users that showed See-Saw at least once and still active after on year: "
      f"{np.count_nonzero(np.logical_and(qualifying_triple_count >= 1, time_between_first_and_last >= timedelta(days=365).total_seconds()))}")


