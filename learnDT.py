# learnDT.py - Learning a binary decision tree
# AIFCA Python3 code Version 0.9.0 Documentation at http://aipython.org

# Artificial Intelligence: Foundations of Computational Agents
# http://artint.info
# Copyright David L Poole and Alan K Mackworth 2017-2020.
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# See: http://creativecommons.org/licenses/by-nc-sa/4.0/deed.en

from learnProblem import Learner, squared_error, absolute_error, log_loss, mean
from learnNoInputs import Predict
import math

class DT_learner(Learner):
    def __init__(self,
                 dataset,
                 split_to_optimize=log_loss,           # to minimize for at each split 
                 leaf_prediction=Predict.mean,   # what to use for point prediction at leaves
                 train=None,                     # used for cross validation
                 min_number_examples=10):
        self.dataset = dataset
        self.target = dataset.target
        self.split_to_optimize = split_to_optimize
        self.leaf_prediction = leaf_prediction
        self.min_number_examples = min_number_examples
        if train is None:
            self.train = self.dataset.train
        else:
            self.train = train

    def learn(self):
        return self.learn_tree(self.dataset.input_features, self.train)
        
    def learn_tree(self, input_features, data_subset):
        """returns a decision tree
        for input_features is a set of possible conditions
        data_subset is a subset of the data used to build this (sub)tree

        where a decision tree is a function that takes an example and
        makes a prediction on the target feature
        """
        if (input_features and len(data_subset) >= self.min_number_examples):
            first_target_val = self.target(data_subset[0])
            allagree = all(self.target(inst)==first_target_val for inst in data_subset)
            if not allagree:
                split, partn = self.select_split(input_features, data_subset)
                if split: # the split succeeded in splitting the data
                    false_examples, true_examples = partn
                    rem_features = [fe for fe in input_features if fe != split]
                    self.display(2,"Splitting on",split.__doc__,"with examples split",
                                   len(true_examples),":",len(false_examples))
                    true_tree = self.learn_tree(rem_features,true_examples)
                    false_tree =  self.learn_tree(rem_features,false_examples)
                    def fun(e):
                        if split(e):
                            return true_tree(e)
                        else:
                            return false_tree(e)
                    #fun = lambda e: true_tree(e) if split(e) else false_tree(e)
                    fun.__doc__ = ("if "+split.__doc__+" then ("+true_tree.__doc__+
                                   ") else ("+false_tree.__doc__+")")
                    return fun
        # don't expand the trees but return a point prediction
        prediction = self.leaf_prediction(self.target(e) for e in data_subset)
        def leaf_fun(e):
            return prediction
        leaf_fun.__doc__ = "{:.7f}".format(prediction)
        return leaf_fun
        
    def select_split(self, input_features, data_subset):
        """finds best feature to split on.

        input_features is a non-empty list of features.
        returns feature, partition
        where feature is an input feature with the smallest error as
              judged by split_to_optimize or
              feature==None if there are no splits that improve the error
        partition is a pair (false_examples, true_examples) if feature is not None
        """
        best_feat = None # best feature
        # best_error = float("inf")  # infinity - more than any error
        best_error = training_error(self.dataset.target, data_subset,
                                        self.split_to_optimize, self.leaf_prediction)
        best_partition = None
        for feat in input_features:
            false_examples, true_examples = partition(data_subset,feat)
            if false_examples and true_examples:  #both partitons are non-empty
                err = (training_error(self.dataset.target, false_examples,
                                          self.split_to_optimize, self.leaf_prediction)
                       + training_error(self.dataset.target, true_examples,
                                            self.split_to_optimize, self.leaf_prediction))
                self.display(3,"   split on",feat.__doc__,"has error=",err,
                          "splits into",len(true_examples),":",len(false_examples))
                if err < best_error:
                    best_feat = feat
                    best_error=err
                    best_partition = false_examples, true_examples
        self.display(3,"best split is on",best_feat.__doc__,
                               "with err=",best_error)
        return best_feat, best_partition

def partition(data_subset,feature):
    """partitions the data_subset by the feature"""
    true_examples = []
    false_examples = []
    for example in data_subset:
        if feature(example):
            true_examples.append(example)
        else:
            false_examples.append(example)
    return false_examples, true_examples


def training_error(target, data_subset, eval_critereon, leaf_prediction):
    """returns training error for dataset on the target (with no more splits)
    We make a single prediction using leaf_prediction
    It is evaluated using eval_critereon for each example
    """
    prediction = leaf_prediction(target(e) for e in data_subset)
    error = sum(eval_critereon(prediction, target(e))
                 for e in data_subset)
    return error

from learnProblem import Data_set, Data_from_file

def testDT(data, print_tree=True, selections = Predict.all):
    """Prints errors and the trees for various evaluation criteria and ways to select leaves.
    """
    evaluation_criteria = [squared_error, absolute_error, log_loss]
    print("Split Choice","Leaf Choice",'\t'.join(ecrit.__doc__
                                                 for ecrit in evaluation_criteria),sep="\t")
    for crit in evaluation_criteria:
        for leaf in selections:
            tree = DT_learner(data, split_to_optimize=crit, leaf_prediction=leaf).learn()
            print(crit.__doc__, leaf.__doc__,
                    "\t".join("{:.7f}".format(data.evaluate_dataset(data.test, tree, ecrit))
                                  for ecrit in evaluation_criteria),sep="\t")
            if print_tree:
                print(tree.__doc__)

if __name__ == "__main__":
    print("SPECT.csv"); testDT(data=Data_from_file('data/SPECT.csv', target_index=0), print_tree=False)
    # print("carbool.csv"); testDT(data = Data_from_file('data/carbool.csv', target_index=-1))
    # print("mail_reading.csv"); testDT(data = Data_from_file('data/mail_reading.csv', target_index=-1))
    # print("holiday.csv"); testDT(data = Data_from_file('data/holiday.csv', num_train=19, target_index=-1))
    
