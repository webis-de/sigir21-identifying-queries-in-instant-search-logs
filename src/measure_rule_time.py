import time

from data.util import *
from rule_based.pipeline import execute_rule_pipeline
from rule_based.steps import time_step, containment_step_with_time, jaccard_similarity_step

FILE = "webis-NIL-21.ldjson"

print("Importing data...")
data_raw = import_data(FILE)

print("Removing single query users...")
data = {}
for user in data_raw.keys():
    if len(data_raw[user]) > 1:
        data[user] = data_raw[user]

pairs = []
for user in data.keys():
    pairs += make_pairs(data[user])

print(len(pairs))
counts = [10_000, 100_000, 500_000]

for count in counts:
    used_pairs = []
    while count > len(used_pairs):
        used_pairs += pairs
    used_pairs = used_pairs[:count]
    start_time = time.time()
    a, b = execute_rule_pipeline(used_pairs, [(time_step, {})])
    end_time_start_containment = time.time()
    c, d = execute_rule_pipeline(used_pairs, [(containment_step_with_time, {})])
    end_containment_start_jaccard = time.time()
    e, f = execute_rule_pipeline(used_pairs, [(jaccard_similarity_step, {"n":3})])
    end_time = time.time()
    print(f"Time for {count}:")
    print(f"Raw: {(end_time - start_time) * 1000} ms")
    print(f"First step (time_gap): {(end_time_start_containment - start_time) / count * 1000} ms")
    print(f"Second step (containment): {(end_containment_start_jaccard - end_time_start_containment)/count * 1000} ms")
    print(f"Third step (lexical_similarity): {(end_time - end_containment_start_jaccard) / count * 1000} ms")