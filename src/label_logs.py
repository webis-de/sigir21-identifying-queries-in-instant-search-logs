import argparse
import pickle

from data.util import import_data, remove_non_ascii_strings, remove_users_with_single_entry, count_previous_occurrence
from features.featuresets import feature_template
from models.training import run_model_behind_rules
from rule_based.rulesets import lexical_dissimilarity

parser = argparse.ArgumentParser(description="Run and validate models on given dataset")
parser.add_argument("dataset", help="Path to dataset")
parser.add_argument("model", help="Path to saved model")
parser.add_argument("output", help="Output path where the final results should be saved.")
parser.add_argument("--dynamic", type=bool, default=False, help="Dynamic lookback enabled")
parser.add_argument("--verbosity", type=int, default=0, help="Level of verbosity")
parser.add_argument("--lookback", type=int, default=3, help="Determins how many log entries our model looks back")
parser.add_argument("--lookahead", type=bool, default=False, help="Enable look ahead")
parser.add_argument("--verbose", type=int, default=0)

args = parser.parse_args()

data = import_data(args.dataset)

print("Removing non-ASCII")
print(f"Users: {len(data.keys())}")
data = remove_non_ascii_strings(data, remove_space=True)
data = remove_users_with_single_entry(data)
print(f"Users after non-ASCII removal: {len(data.keys())}")

count_previous_occurrence(data)

# run model
model = pickle.load(open(args.model, "rb"), fix_imports=True)

model_results = run_model_behind_rules(model=model,
                                       users=data,
                                       rule_set=lexical_dissimilarity,
                                       feature_set_template=feature_template,
                                       dynamic=args.dynamic,
                                       static_lookback=args.lookback,
                                       verbose=args.verbose)
# export users
with open(args.output, "w") as outputfile:
    for user_key in data.keys():
        for entry in data[user_key]:
            outputfile.write(entry.create_json(boundary=True))
            outputfile.write("\n")



