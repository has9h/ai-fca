# learnNoInputs.py - Learning ignoring all input features
# AIFCA Python3 code Version 0.9.0 Documentation at http://aipython.org

# Artificial Intelligence: Foundations of Computational Agents
# http://artint.info
# Copyright David L Poole and Alan K Mackworth 2017-2020.
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# See: http://creativecommons.org/licenses/by-nc-sa/4.0/deed.en

from learnProblem import squared_error, absolute_error, log_loss, mean
import math, random, statistics
import utilities  # argmax for (element,value) pairs

class Predict(object):
    """The class of prediction methods for a list of numbers
    Please make the doc strings the same length, because they are used in tables.
    Note that we don't need self argument, as we are creating Predict objects,
    To use call Predict.laplace(data) etc."""

    def mean(data):
        "mean        "
        return mean(data)
    
    def bounded_mean(data, bound=0.01):
        "bounded mean"
        return min(max(mean(data),bound),1-bound)

    def laplace(data):
        "Laplace     "  # for Boolean (or 0/1 data only)
        return mean(data, isum=1, icount=2)

    def mode(data):
        "mode        "
        counts = {}
        for e in data:
            if e in counts:
                counts[e] += 1
            else:
                counts[e] = 1
        return utilities.argmaxe(counts.items())

    def median(data):  
        "median      "
        return statistics.median(data)

    all = [mean, bounded_mean, laplace, mode, median]

def evaluate(train_size, predictor, error_measure, num_samples=10000, test_size=10 ):
    """return the average error when
   train_size is the number of training examples
   predictor(training) -> [0,1]
   error_measure(prediction,actual) -> non-negative reals
   """
    error = 0
    for sample in range(num_samples):
        prob = random.random()
        training = [1 if random.random()<prob else 0 for i in range(train_size)]
        prediction = predictor(training)
        test = (1 if random.random()<prob else 0 for i in range(test_size))
        error += sum( error_measure(prediction,actual) for actual in test)/test_size
    return error/num_samples

def test_no_inputs(error_measures = [squared_error, absolute_error, log_loss]):
    for train_size in [1,2,3,4,5,10,20,100,1000]:
        print("For training size",train_size,":")
        print("   Predictor","\t".join(error_measure.__doc__ for
                                           error_measure in error_measures),sep="\t")
        for predictor in Predict.all:
            print(f"   {predictor.__doc__}",
                      "\t".join("{:.7f}".format(evaluate(train_size, predictor, error_measure))
                                    for error_measure in error_measures),sep="\t")
        
if __name__ == "__main__":
    test_no_inputs()
        
