import argparse

from data.util import import_data

parser = argparse.ArgumentParser(description="Run and validate models on given dataset")
parser.add_argument("infile", help="Path splitted log")
parser.add_argument("outfile", help="Path to save query file")
args = parser.parse_args()

print("Loading splitted log...")
splitted_log = import_data(args.infile)

print("Extracting queries...")
queries = {}
for user in splitted_log:
    for query in splitted_log[user]:
        if query.boundary:
            if user in queries:
                queries[user].append(query)
            else:
                queries[user] = [query]

print("Saving queries...")
with open(args.outfile, "w") as file:
    for user in queries:
        for query in queries[user]:
            file.write(query.create_json(boundary=True))
            file.write("\n")

