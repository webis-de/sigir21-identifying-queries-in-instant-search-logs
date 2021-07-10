# Setup

run `pip3 install -r requirements.txt`

# Overview

* `run_models.py` main file for training and validation; see below
* `measure_rule_time.py` measure run time for single rules
* `remove_bots.py` remove bot users
* `label_log.py` split a log into queries with the trained model
* `keep_only_queries.py` keep only the query (the last log entry before a split)
* `log_insight.py` basic statistics for splitted log
* `see_saw_stats.py` statisics for see-saw behavior in splitted log
 

# Paper results

## Time measurements

Specify path to data in variable `FILE` in file `measure_rule_time.py`and run it.
The run time results for 

## Accuracy results
General usage:
"python run_models [path_to_dataset]"
The results of the paper could be reconstructed by using the parameter "--test=True"
In this case all results are determined on the test set (with the same parameters used as in the paper).

For further examination of the errors it is possible to save all false positives, false negative and true positive pairs.
To do this use the parameter "--savefp=True", "--savefn=True" or "--savet**p=True"

### Rules
For calculating the results after the rules it is necessary to specify the parameter '--model=none'. 
This assures that no model is called and all results are the results of the rules.

The output of the rules is in the following form.

    Total pairs: n
    Predicted pairs: m
    Percent decided: m/n
    Metrics for this pairs:
    Results: ACC =  PRE =  REC =  F2 =  TP =  TN =  FP =  FN = 
    Needed time: ms
    Time per pair:  ms

The predicted pairs are those pairs which could be decided with enough certainty.

Each rule has pairs which could not decided with enough certainty. 
In our proposed method those pairs are decided with machine learning. 
To determine the accuracy values of the rules only we could set the remaining (undecided) pairs to the
opposite of the rule decision (i.e. if the rule is designed to split pairs all uncertain entries are merged together). 
To do this, supply the parameter "--finalrules=True"
Keep in mind in this case the percentage of decided pairs is misleading. 

For the paper results the following rulesets were used

1. time_gap - split after 5 minutes
2. containment - check for string containment after the 5 minutes split
3. lexical_similarity - for undecided pairs invoke the lexical similarity step
4. lexical_dissimilarity - for undecided pairs invoke the lexical dissimilarity step 
5. cetendil - decision by edit distance by Centdil et al. 
6. kim - decision by edit distance and time by Kim et al.
7. hagen - steps 1 to 3 of cascading log segmentation by Hagen et al.

    python3 run_models [path_to_dataset] --model=none --rules=time --test=True
    python3 run_models [path_to_dataset] --model=none --rules=time --finalrules=True --test=True
    python3 run_models [path_to_dataset] --model=none --rules=containment --test=True
    python3 run_models [path_to_dataset] --model=none --rules=containment --finalrules=True --test=True
    python3 run_models [path_to_dataset] --model=none --rules=lexical_similarity --test=True
    python3 run_models [path_to_dataset] --model=none --rules=lexical_similarity --finalrules=True --test=True
    python3 run_models [path_to_dataset] --model=none --rules=lexical_dissimilarity --test=True
    python3 run_models [path_to_dataset] --model=none --rules=lexical_dissimilarity --finalrules=True --test=True

### Model

For training the model behind a set of rules the following command can be used:

    python3 run_models [path_to_dataset] --model=lgr --rules=[rule_set] --train=True

Of course the rule set used in the paper is lexical_dissimilarity.
This command runs each pair through the rules of the ruleset and if it is still undecided it will be used for training of the logistic regression.

After training a model the results can be obtained by using 
    
    python3 run_models [path_to_dataset] --model=lgr --rules=[rule_set] --modelpath=[path to the model] --test=True

The pre trained model used in the paper is attached too. Its file name is: `trained_model_paper.pyc`






